# coding=utf-8

from BaseTasksFormatter import *

class RedmineFormatter(BaseTasksFormatter):
    def __init__(self, silent, issues_generator):
        BaseTasksFormatter.__init__(self, silent, issues_generator)

    def format_tasks(self, edges):
        result_lines = []
        print_data = self.prepate_print_data(edges)

        for version, parents in print_data.iteritems():
            result_lines.append('* %s' % version)
            version_lines = []
            for parent, issues in parents.iteritems():
                for issue in issues:
                    task = issue['issue']
                    version_lines.append(
                        ((u'** "%s":%s – %s (%f)' % (
                            task.id, task.url, task.issue_name, issue['weight'])), issue['weight'])
                    )
            version_lines = sorted(version_lines, key=lambda x: x[1], reverse=True)
            for line, weight in version_lines:
                result_lines.append(line)
        return u'\n'.join(result_lines).encode('utf-8')