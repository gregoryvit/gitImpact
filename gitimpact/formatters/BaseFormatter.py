from issues import *


class BaseFormatter(object):
    def __init__(self, silent):
        self.silent = silent

    def format_tasks(self, source_tasks, commits, features, task_commits, result_tasks, result_edges):
        return ""
