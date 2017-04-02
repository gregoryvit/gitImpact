from BaseFormatter import *

import graphviz


class StrictDigraph(graphviz.dot.Dot):
    """Directed graph source code in the DOT language."""

    __doc__ += graphviz.Graph.__doc__.partition('.')[2]

    _head = 'strict digraph %s{'
    _edge = '\t\t%s -> %s%s'
    _edge_plain = '\t\t%s -> %s'


class GraphvizFormatter(BaseFormatter):
    def __init__(self, silent, output_filepath, output_format):
        BaseFormatter.__init__(self, silent)
        self.output_filepath = output_filepath
        self.output_format = output_format

    def format_tasks(self, source_tasks, commits, features, task_commits, result_tasks, result_edges):
        dot = StrictDigraph(comment='', engine='dot', format=self.output_format)
        dot.graph_attr['rankdir'] = 'LR'

        for feature, commits_dicts in features.iteritems():
            tasks = [task_commits[commits_dict["commit"]] for commits_dict in commits_dicts]
            flatten_tasks = [task for sub_list in tasks for task in sub_list]
            for task in flatten_tasks:
                dot.edge(feature, task.str_id)

        for commit, commit_features in commits.iteritems():
            for feature_dict in commit_features:
                dot.edge(commit, feature_dict["file_path"])

        for task, task_commits in source_tasks.iteritems():
            for commit in task_commits:
                dot.edge(task, commit)

        for task, weight in result_edges:
            dot.node(task.str_id, "%s (%f)" % (task.str_id, weight))

        dot.render(self.output_filepath, view=False)

        return ""
