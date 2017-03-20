from BaseFormatter import *

import graphviz

class StrictDigraph(graphviz.dot.Dot):
    """Directed graph source code in the DOT language."""

    __doc__ += graphviz.Graph.__doc__.partition('.')[2]

    _head = 'strict digraph %s{'
    _edge = '\t\t%s -> %s%s'
    _edge_plain = '\t\t%s -> %s'

class GraphvizFormatter(BaseFormatter):
    def __init__(self, silent, output_filepath):
        BaseFormatter.__init__(self, silent)
        self.output_filepath = output_filepath

    def format_tasks(self, source_tasks, commits, features, result_tasks):
        dot = StrictDigraph(comment='The Round Table', engine='dot', format='svg')
        dot.graph_attr['rankdir'] = 'LR'

        for feature, tasks in features.iteritems():
            for task in tasks:
                dot.edge(feature, task.str_id)

        for commit, commit_features in commits.iteritems():
            for feature in commit_features:
                dot.edge(commit, feature)

        for task, task_commits in source_tasks.iteritems():
            for commit in task_commits:
                dot.edge(task, commit)

        for task, weight  in result_tasks:
            dot.node(task.str_id, "%s (%f)" % (task.str_id, weight))

        dot.render(self.output_filepath, view=False)

        return ""