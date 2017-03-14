from issues import *

class BaseTasksFormatter(object):
    def __init__(self, silent, issues_generator):
        self.silent = silent
        self.issues_generator = issues_generator

    def prepate_print_data(self, edges):
        print_data = {}

        items = [(idx, edge_info[0], edge_info[1]) for idx, edge_info in enumerate(edges)]
        # tasks = [task for _, task, _ in items]

        # task_ids = BugTrackerIssue.load_issues(tasks, tasks[0].redmine_api)
        # items = [item for item in items if int(item[1].id) in task_ids]
        # print items
        for idx, task, weight in items:
            issue = self.issues_generator.create_issue(task.raw_id)
            try:
                issue.load_data()

                issue_data = {
                    'task': task,
                    'issue': issue,
                    'weight': weight
                }

                version_name = issue.version_name if issue.version_name is not None else ''
                parent_name = issue.parent_name if issue.parent_name is not None else ''

                if version_name in print_data:
                    version_data = print_data[version_name]

                    if parent_name in version_data:
                        parent_data = version_data[parent_name]
                        parent_data.append(issue_data)
                        version_data[parent_name] = parent_data
                    else:
                        version_data[parent_name] = [issue_data]

                    print_data[version_name] = version_data
                else:
                    print_data[version_name] = {parent_name: [issue_data]}

                if not self.silent:
                    print "%s (%d/%d)" % (issue.url, idx + 1, len(edges))
            except Exception as e:
                if not self.silent:
                    print "%s (%d/%d) # %s" % (issue.url, idx + 1, len(edges), e)

        return print_data

    def format_tasks(self, edges):
        return ""