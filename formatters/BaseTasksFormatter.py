class BaseTasksFormatter(object):
    def __init__(self, silent):
        self.silent = silent

    def prepate_print_data(self, edges):
        print_data = {}

        for idx, edge_info in enumerate(edges):
            task, weight = edge_info
            try:
                task.load_data()

                issue_data = {
                    'task': task,
                    'weight': weight
                }

                version_name = task.version_name if task.version_name is not None else ''
                parent_name = task.parent_name if task.parent_name is not None else ''

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
                    print "%s (%d/%d)" % (task.url, idx + 1, len(edges))
            except Exception as e:
                if not self.silent:
                    print "%s (%d/%d) # %s" % (task.url, idx + 1, len(edges), e)

        return print_data

    def format_tasks(self, edges):
        return ""