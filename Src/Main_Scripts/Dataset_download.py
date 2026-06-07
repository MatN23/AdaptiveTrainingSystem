# Copyright (c) 2025 MatN23. All rights reserved.
# Licensed under the Custom License below.

import os
import logging
import json
from typing import Optional, Dict, List, Tuple
from pathlib import Path
from collections import defaultdict

try:
    from datasets import load_dataset
except ImportError:
    print("Error: 'datasets' library not found. Install with: pip install datasets")
    print("Run: pip install datasets huggingface_hub")
    exit(1)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# CONFIGURATION
MAX_FILE_SIZE_MB = 100  # Maximum file size in MB
DATASET_NAME = "OpenAssistant/oasst2"  # Using OASST2

def setup_output_directory(project_root: Optional[str] = None) -> Path:
    """Setup and create output directory for dataset files."""
    if project_root is None:
        script_dir = Path(__file__).parent.absolute()
        project_root = script_dir
    
    # Keep the same path as before (oasst1_data)
    output_dir = Path(project_root) / "oasst1_data"
    
    # Ensure the directory exists
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f" Created output directory: {output_dir}")
    except Exception as e:
        logger.error(f" Failed to create output directory: {e}")
        raise
    
    logger.info(f"Project root: {project_root}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f" Max file size: {MAX_FILE_SIZE_MB}MB per file")
    
    return output_dir

def build_conversation_tree(messages: List[Dict]) -> Tuple[Dict[str, Dict], List[str]]:
    """Build a tree structure from messages using parent-child relationships."""
    message_map = {}
    
    # First pass: create message map
    for msg in messages:
        message_map[msg['message_id']] = {
            'data': msg,
            'children': []
        }
    
    # Second pass: build parent-child relationships
    root_messages = []
    for msg in messages:
        parent_id = msg.get('parent_id')
        if parent_id and parent_id in message_map:
            message_map[parent_id]['children'].append(msg['message_id'])
        else:
            # This is a root message (conversation starter)
            root_messages.append(msg['message_id'])
    
    return message_map, root_messages

def extract_conversation_paths(message_map: Dict, root_id: str) -> List[List[Dict]]:
    """Extract all possible conversation paths from a root message."""
    conversations = []
    
    def dfs_path(node_id: str, current_path: List[Dict]):
        if node_id not in message_map:
            return
        
        node = message_map[node_id]
        new_path = current_path + [node['data']]
        
        # Save conversation at multiple points to generate more data
        if len(new_path) >= 2:  # At least one exchange
            conversations.append(new_path.copy())
        
        # If this is a leaf node or has no valid children, we're done with this path
        if not node['children']:
            return
        else:
            # Continue down each child path
            for child_id in node['children']:
                dfs_path(child_id, new_path)
    
    dfs_path(root_id, [])
    return conversations

def format_conversation(messages: List[Dict]) -> Dict:
    """Format a conversation path into a structured format."""
    conversation = {
        'conversation_id': messages[0].get('message_tree_id', ''),
        'messages': [],
        'total_turns': len(messages),
        'languages': list(set(msg.get('lang', 'en') for msg in messages)),
        'created_date': messages[0].get('created_date', ''),
        'tree_state': messages[0].get('tree_state', '')
    }
    
    for i, msg in enumerate(messages):
        formatted_msg = {
            'turn': i + 1,
            'role': msg.get('role', '').lower(),
            'content': msg.get('text', '').strip(),
            'message_id': msg.get('message_id', ''),
            'review_result': msg.get('review_result', None),
            'rank': msg.get('rank', 0),
            'synthetic': msg.get('synthetic', False),
            'model_name': msg.get('model_name', '')
        }
        conversation['messages'].append(formatted_msg)
    
    return conversation

def filter_quality_conversations(conversations: List[Dict], strict_filtering: bool = False) -> List[Dict]:
    """Filter conversations based on quality criteria."""
    quality_conversations = []
    
    for conv in conversations:
        # Quality checks
        valid = True
        
        # Must have at least 2 turns (one exchange)
        if len(conv['messages']) < 2:
            continue
            
        # Must start with a prompter (user)
        if conv['messages'][0]['role'] != 'prompter':
            continue
            
        # Must alternate between prompter and assistant
        expected_role = 'prompter'
        for msg in conv['messages']:
            if msg['role'] != expected_role:
                valid = False
                break
            expected_role = 'assistant' if expected_role == 'prompter' else 'prompter'
        
        if not valid:
            continue
        
        # Very relaxed filtering to maximize dataset size
        has_content = True
        for msg in conv['messages']:
            # Only filter out completely empty messages
            if not msg['content'] or len(msg['content'].strip()) == 0:
                has_content = False
                break
                
        if not has_content:
            continue
            
        quality_conversations.append(conv)
    
    return quality_conversations

def analyze_conversations(conversations: List[Dict], split_name: str):
    """Analyze the structure and quality of conversations."""
    if not conversations:
        logger.warning(f"No conversations found in {split_name}")
        return
    
    total_conversations = len(conversations)
    total_turns = sum(len(conv['messages']) for conv in conversations)
    avg_turns = total_turns / total_conversations if total_conversations > 0 else 0
    
    turn_distribution = defaultdict(int)
    role_counts = defaultdict(int)
    
    for conv in conversations:
        turn_distribution[len(conv['messages'])] += 1
        for msg in conv['messages']:
            role_counts[msg['role']] += 1
    
    logger.info(f" Conversation Analysis for {split_name}:")
    logger.info(f"  Total conversations: {total_conversations:,}")
    logger.info(f"  Total turns: {total_turns:,}")
    logger.info(f"  Average turns per conversation: {avg_turns:.1f}")
    logger.info(f"  Turn distribution (top 10):")
    
    for turns, count in sorted(turn_distribution.items())[:10]:
        logger.info(f"    {turns} turns: {count:,} conversations ({count/total_conversations*100:.1f}%)")
    
    logger.info(f"  Role distribution:")
    for role, count in role_counts.items():
        logger.info(f"    {role}: {count:,} messages ({count/total_turns*100:.1f}%)")

def get_file_size_mb(file_path: Path) -> float:
    """Get file size in MB."""
    if not file_path.exists():
        return 0
    return file_path.stat().st_size / (1024 * 1024)

def save_conversations_with_size_limit(
    conversations: List[Dict], 
    output_dir: Path, 
    split_name: str,
    max_size_mb: int = 100
) -> List[Path]:
    """
    Save conversations to files, splitting into multiple files if needed to stay under size limit.
    
    Returns:
        List of file paths created
    """
    if not conversations:
        logger.warning(f"No conversations to save for {split_name}")
        return []
    
    saved_files = []
    file_index = 0
    current_file_convs = []
    current_size_estimate = 0
    
    # Estimate average size per conversation (in bytes)
    sample_conv = json.dumps(conversations[0], ensure_ascii=False) + '\n'
    avg_conv_size = len(sample_conv.encode('utf-8'))
    max_size_bytes = max_size_mb * 1024 * 1024
    
    logger.info(f" Estimated average conversation size: {avg_conv_size / 1024:.1f} KB")
    logger.info(f" Target file size: <{max_size_mb}MB")
    
    for i, conv in enumerate(conversations):
        conv_json = json.dumps(conv, ensure_ascii=False) + '\n'
        conv_size = len(conv_json.encode('utf-8'))
        
        # Check if adding this conversation would exceed the limit
        if current_size_estimate + conv_size > max_size_bytes and current_file_convs:
            # Save current batch
            output_file = output_dir / f"oasst1_{split_name}.jsonl" if file_index == 0 else output_dir / f"oasst1_{split_name}_part{file_index+1}.jsonl"
            logger.info(f" Saving batch {file_index + 1}: {len(current_file_convs):,} conversations (~{current_size_estimate / (1024*1024):.1f}MB)")
            
            with open(output_file, 'w', encoding='utf-8') as f:
                for c in current_file_convs:
                    f.write(json.dumps(c, ensure_ascii=False) + '\n')
            
            actual_size = get_file_size_mb(output_file)
            logger.info(f" Saved {output_file.name}: {actual_size:.1f}MB, {len(current_file_convs):,} conversations")
            saved_files.append(output_file)
            
            # Reset for next batch
            file_index += 1
            current_file_convs = []
            current_size_estimate = 0
        
        # Add conversation to current batch
        current_file_convs.append(conv)
        current_size_estimate += conv_size
        
        # Progress update
        if (i + 1) % 1000 == 0:
            logger.info(f"  Processed {i+1:,}/{len(conversations):,} conversations...")
    
    # Save remaining conversations
    if current_file_convs:
        output_file = output_dir / f"oasst1_{split_name}.jsonl" if file_index == 0 else output_dir / f"oasst1_{split_name}_part{file_index+1}.jsonl"
        logger.info(f" Saving final batch {file_index + 1}: {len(current_file_convs):,} conversations (~{current_size_estimate / (1024*1024):.1f}MB)")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for c in current_file_convs:
                f.write(json.dumps(c, ensure_ascii=False) + '\n')
        
        actual_size = get_file_size_mb(output_file)
        logger.info(f" Saved {output_file.name}: {actual_size:.1f}MB, {len(current_file_convs):,} conversations")
        saved_files.append(output_file)
    
    return saved_files

def download_and_process_conversations(output_dir: Path) -> bool:
    """Download OASST2 dataset and process into conversation format with size limits."""
    try:
        logger.info(f" Loading OpenAssistant dataset ({DATASET_NAME})...")
        logger.info("This may take a few minutes for the first download...")
        
        # Load dataset with error handling
        try:
            ds = load_dataset(DATASET_NAME, trust_remote_code=True)
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            logger.info(" Possible solutions:")
            logger.info("  1. Try running: huggingface-cli login")
            logger.info("  2. Install git-lfs if not installed: sudo apt-get install git-lfs")
            logger.info("  3. Check your internet connection")
            return False
        
        logger.info(" Dataset loaded successfully!")
        logger.info(f" Available splits: {list(ds.keys())}")
        
        # Process train and validation splits
        splits_to_process = ['train', 'validation']
        
        # Check if test/eval split exists
        if 'test' in ds:
            splits_to_process.append('test')
        
        all_saved_files = []
        
        for split_name in splits_to_process:
            if split_name not in ds:
                logger.warning(f"Split '{split_name}' not found in dataset")
                continue
                
            logger.info(f"\n{'='*70}")
            logger.info(f" Processing {split_name} split...")
            logger.info(f"{'='*70}")
            
            # Get split data
            split_data = ds[split_name]
            logger.info(f" Total messages in {split_name}: {len(split_data):,}")
            
            # Filter for messages
            valid_messages = []
            
            for i, msg in enumerate(split_data):
                # Show progress for large datasets
                if i > 0 and i % 10000 == 0:
                    logger.info(f"  Processed {i:,}/{len(split_data):,} messages...")
                
                # Very basic validation
                if (msg.get('text', '').strip() and  # Has some content
                    msg.get('message_tree_id')):  # Has tree ID
                    valid_messages.append(msg)
            
            logger.info(f" Valid messages in {split_name}: {len(valid_messages):,}")
            
            # Group messages by conversation tree
            tree_messages = defaultdict(list)
            for msg in valid_messages:
                tree_id = msg.get('message_tree_id', '')
                if tree_id:
                    tree_messages[tree_id].append(msg)
            
            logger.info(f" Conversation trees in {split_name}: {len(tree_messages):,}")
            
            # Build conversations
            all_conversations = []
            processed_trees = 0
            
            for tree_id, messages in tree_messages.items():
                processed_trees += 1
                if processed_trees % 1000 == 0:
                    logger.info(f"  Processing tree {processed_trees:,}/{len(tree_messages):,}...")
                
                # Build conversation tree
                message_map, root_messages = build_conversation_tree(messages)
                
                # Extract conversation paths from each root
                for root_id in root_messages:
                    paths = extract_conversation_paths(message_map, root_id)
                    for path in paths:
                        conv = format_conversation(path)
                        all_conversations.append(conv)
            
            logger.info(f" Raw conversations extracted from {split_name}: {len(all_conversations):,}")
            
            # Apply filtering
            filtered_conversations = filter_quality_conversations(all_conversations, strict_filtering=False)
            logger.info(f" Filtered conversations: {len(filtered_conversations):,}")
            
            # Analyze dataset
            logger.info(f"\n--- Analysis for {split_name} ---")
            analyze_conversations(filtered_conversations, split_name)
            
            # Save with size limit
            if not filtered_conversations:
                logger.warning(f"No conversations to save for {split_name}")
                continue
            
            logger.info(f"\n Saving {split_name} conversations with {MAX_FILE_SIZE_MB}MB size limit...")
            saved_files = save_conversations_with_size_limit(
                filtered_conversations,
                output_dir,
                split_name,
                max_size_mb=MAX_FILE_SIZE_MB
            )
            
            all_saved_files.extend(saved_files)
            
            logger.info(f" {split_name} split complete: {len(saved_files)} file(s) created")
        
        logger.info(f"\n{'='*70}")
        logger.info(f" All splits processed: {len(all_saved_files)} total files")
        logger.info(f"{'='*70}")
        
        return True
        
    except Exception as e:
        logger.error(f" Error processing conversations: {e}")
        import traceback
        traceback.print_exc()
        return False

def validate_conversation_files(output_dir: Path) -> bool:
    """Validate that conversation files exist and are readable."""
    # Get all conversation files in the directory
    conversation_files = list(output_dir.glob("oasst1_*.jsonl"))
    
    if not conversation_files:
        logger.error(" No conversation files found")
        return False
    
    for file_path in conversation_files:
        if not file_path.exists():
            logger.error(f" Missing file: {file_path}")
            return False
        
        file_size_mb = get_file_size_mb(file_path)
        
        if file_size_mb == 0:
            logger.error(f" Empty file: {file_path}")
            return False
        
        if file_size_mb > MAX_FILE_SIZE_MB:
            logger.warning(f"  File exceeds {MAX_FILE_SIZE_MB}MB limit: {file_path} ({file_size_mb:.1f}MB)")
        
        # Test reading and parsing
        try:
            line_count = 0
            with open(file_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    line_count += 1
                    
                    # Test parsing of first few lines
                    if i < 3:
                        line = line.strip()
                        if not line:
                            continue
                        
                        try:
                            data = json.loads(line)
                            # Check conversation structure
                            required_fields = ['conversation_id', 'messages', 'total_turns']
                            for field in required_fields:
                                if field not in data:
                                    logger.error(f" Missing field '{field}' in {file_path}")
                                    return False
                            
                            # Check message structure
                            if data['messages']:
                                msg = data['messages'][0]
                                msg_fields = ['role', 'content', 'turn']
                                for field in msg_fields:
                                    if field not in msg:
                                        logger.error(f" Missing message field '{field}' in {file_path}")
                                        return False
                                        
                        except json.JSONDecodeError as e:
                            logger.error(f" Invalid JSON in {file_path} line {i+1}: {e}")
                            return False
            
            logger.info(f" {file_path.name}: {file_size_mb:.1f}MB, {line_count:,} conversations validated")
                        
        except Exception as e:
            logger.error(f" Cannot read file {file_path}: {e}")
            return False
    
    logger.info(" All conversation files validated successfully!")
    return True

def check_existing_files(output_dir: Path) -> bool:
    """Check if dataset files already exist and are valid."""
    # Look for any existing conversation files
    existing_files = list(output_dir.glob("oasst1_*.jsonl"))
    
    if existing_files:
        logger.info(" Found existing conversation files!")
        for file_path in existing_files:
            size_mb = get_file_size_mb(file_path)
            logger.info(f"  - {file_path.name}: {size_mb:.1f} MB")
        
        logger.info(" Validating existing files...")
        return validate_conversation_files(output_dir)
    
    return False

def main():
    """Main function to download and prepare OASST2 conversational dataset."""
    logger.info(" Starting OASST2 Conversational Dataset Download...")
    logger.info(f" Dataset: {DATASET_NAME}")
    logger.info(f" Max file size: {MAX_FILE_SIZE_MB}MB")
    logger.info("=" * 70)
    
    try:
        # Setup directories
        output_dir = setup_output_directory()
        
        # Check if files already exist and are valid
        if check_existing_files(output_dir):
            logger.info(" Valid dataset files already exist!")
            logger.info(" Delete files in the output directory if you want to re-download")
            logger.info("=" * 70)
            return 0
        else:
            logger.info(" Downloading/reprocessing dataset...")
        
        # Download and process conversations
        success = download_and_process_conversations(output_dir)
        
        if not success:
            logger.error(" Dataset processing failed!")
            return 1
        
        # Validate files
        if not validate_conversation_files(output_dir):
            logger.error(" File validation failed!")
            return 1
        
        # Success summary
        logger.info("=" * 70)
        logger.info(" Conversational dataset preparation completed!")
        logger.info(f" Files saved in: {output_dir}")
        logger.info(f" Dataset source: {DATASET_NAME}")
        logger.info("")
        logger.info(" Generated Files:")
        
        # List all generated files with sizes
        total_size = 0
        total_convs = 0
        for file_path in sorted(output_dir.glob("oasst1_*.jsonl")):
            size_mb = get_file_size_mb(file_path)
            line_count = sum(1 for _ in open(file_path, 'r', encoding='utf-8'))
            total_size += size_mb
            total_convs += line_count
            status = "" if size_mb <= MAX_FILE_SIZE_MB else ""
            logger.info(f"   {status} {file_path.name}: {size_mb:.1f}MB, {line_count:,} conversations")
        
        logger.info("")
        logger.info(f" Total: {total_size:.1f}MB across {total_convs:,} conversations")
        logger.info("")
        logger.info(" Dataset Format:")
        logger.info("   Each line contains a complete conversation with:")
        logger.info("   - conversation_id: Unique identifier")
        logger.info("   - messages: Array of turn-by-turn exchanges")
        logger.info("   - total_turns: Number of messages in conversation")
        logger.info("   - Each message has: role, content, turn number")
        logger.info("")
        logger.info(" Ready for conversational training!")
        logger.info("=" * 70)
        
        return 0
        
    except KeyboardInterrupt:
        logger.info(" Processing interrupted by user")
        return 1
    except Exception as e:
        logger.error(f" Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())