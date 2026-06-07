from colossalqa.text_splitter.chinese_text_splitter import ChineseTextSplitter


def test_text_splitter():
    # unit test
    spliter = ChineseTextSplitter(chunk_size=30, chunk_overlap=0)
    out = spliter.split_text(
        "4FSMNCTC750KFbankchartokenbasetrain + finetunebasetrain1basetrainfinetune"
    )
    print(len(out))
    assert len(out) == 4  # ChineseTextSplitter will not break sentence. Hence the actual chunk size is not 30