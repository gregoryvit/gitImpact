# coding=utf-8

from BaseTasksFormatter import *

class FriendlyFormatter(BaseTasksFormatter):
    def __init__(self, silent, issues_generator):
        BaseTasksFormatter.__init__(self, silent, issues_generator)

    def format_tasks(self, edges):
        result_lines = []
        print_data = self.prepate_print_data(edges)

        for version, parents in print_data.iteritems():
            result_lines.append(version)
            for parent, issues in parents.iteritems():
                result_lines.append(u'\t%s' % parent)
                sorted_issues = sorted(issues, key=lambda x: x['weight'], reverse=True)
                for issue in sorted_issues:
                    task = issue['issue']
                    result_lines.append(
                        (u'\t\t* %s – %s (%f)' % (task.issue_name, task.url, issue['weight'])))
        return u'\n'.join(result_lines).encode('utf-8')