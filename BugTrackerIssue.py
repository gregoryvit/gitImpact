from redmine import Redmine

class BugTrackerIssue():
    def __init__(self, id, host, key):
        self.id = id
        self.redmine_api = Redmine('https://%s' % host, key=key)
        self.parent_name = None
        self.issue_name = None
        self.version_name = None
        self.url = 'https://%s/issues/%s' % (host, id)

    def load_data(self):
        self.parent_name = None
        self.issue_name = None
        self.version_name = None

        issue = self.redmine_api.issue.get(self.id)
        self.fill_data(issue)

    def fill_data(self, issue):
        self.issue_name = issue.subject

        if issue.parent.id is not None:
            parent_issue = self.redmine_api.issue.get(issue.parent.id)
            self.parent_name = parent_issue.subject
            self.version_name = issue.fixed_version.name

    @staticmethod
    def load_issues(issues, redmine_api):
        ids = [issue.id for issue in issues]
        # print redmine_api.issue.filter(tuple(ids)).total_count
        import time
        start = time.time()
        if len(ids) > 90:
            str_ids = ','.join(ids)
            int_ids = map(int, ids)
            first_issue = redmine_api.issue.get(int_ids[0])
            project_id = first_issue.project.id
            all_issues = redmine_api.issue.filter(status_id="*", project_id=project_id, limit=1000, issue_id=str_ids).filter(tuple(int_ids))
        # else:
        #     all_issues = []
        #     for issue_id in ids:
        #         try:
        #             all_issues.append(redmine_api.issue.get(int(issue_id)))
        #         except:
        #             continue
        else:
            from multiprocessing.pool import ThreadPool

            def foo(issue_id):
                try:
                    return redmine_api.issue.get(int(issue_id))
                except:
                    return None

            pool = ThreadPool(10)
            all_issues = []
            all_issues = pool.map(foo,ids)

            pool.close()
            pool.join()
            try:
                all_issues = [r.get() for r in all_issues]
            except:
                pass
        # print time.time() - start
        # print len(all_issues)
        result_issues = []
        for issue in issues:
            for result_issue in all_issues:
                if result_issue is None or int(issue.id) != int(result_issue.id):
                    continue
                issue.fill_data(result_issue)
                result_issues.append(issue)
        return [issue.id for issue in result_issues]
