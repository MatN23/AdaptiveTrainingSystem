# Copyright (c) 2025 MatN23. All rights reserved.
class LayoutException(Exception):
    pass


class DuplicatedShardingDimensionError(LayoutException):
    pass


class ShardingNotDivisibleError(LayoutException):
    pass


class ShardingOutOfIndexError(LayoutException):
    pass