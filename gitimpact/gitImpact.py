# coding=utf-8

import core
from optparse import OptionParser
import os


def parse_yaml(path_to_yaml_file):
    import ruamel.yaml

    with open(path_to_yaml_file, 'r') as yaml_file:
        try:
            return ruamel.yaml.load(yaml_file, Loader=ruamel.yaml.Loader)
        except ruamel.yaml.YAMLError as exc:
            print(exc)


def get_option(option, config, alt=None, default=None):
    if alt is not None and alt != default:
        return alt

    if option in config:
        return config[option]

    return default


def get_formatters(formatter_dict, silent):
    def get_redmine_generator(key):
        if "issues_generator" in formatter_dict[key]:
            from issues.redmine import RedmineIssuesGenerator

            generator_options = formatter_dict[key]["issues_generator"]
            return RedmineIssuesGenerator(generator_options["host"], generator_options["api_key"])

    result_formatters = []
    for formatter_key, formatter_options in formatter_dict.iteritems():
        if formatter_key == "graphviz_formatter":
            result_formatters.append(core.GraphvizFormatter(silent, **formatter_options))
        elif formatter_key == "redmine_formatter":
            formatter_options["issues_generator"] = get_redmine_generator(formatter_key)
            result_formatters.append(core.RedmineFormatter(silent, **formatter_options))
        elif formatter_key == "friendly_formatter":
            formatter_options["issues_generator"] = get_redmine_generator(formatter_key)
            result_formatters.append(core.FriendlyFormatter(silent, **formatter_options))
    return result_formatters


def main(app_options):
    yaml_config = parse_yaml(app_options.yaml_config_file)

    task_format = get_option("task_format", yaml_config, default={})

    result_options = {
        "excluded_tasks": get_option("excluded_tasks", yaml_config, default=[]),
        "exclude_features": get_option("exclude_features", yaml_config, default=[]),
        "output_filepath": get_option("output_filepath", yaml_config),
        "task_id": get_option("task_id", yaml_config, alt=app_options.task_id),
        "commits": get_option("commits", yaml_config, alt=app_options.commit, default=[]),
        "min_weight": float(get_option("min_weight", yaml_config, alt=app_options.min_weight, default=0.1)),
        "min_impact_rate": get_option("min_impact_rate", yaml_config, alt=app_options.min_impact_rate, default=0.15),
        "silent": get_option("silent", yaml_config, alt=app_options.silent, default=False),
        "source_dir": app_options.repo_path,
        "task_format": (
            get_option("output_format", task_format, default=""), get_option("parse_regex", task_format, default=""))
    }

    check_all_commits = get_option("check_all_commits", yaml_config, alt=app_options.check_all_commits, default=False)
    result_options["check_only_child_commits"] = (not bool(check_all_commits))
    result_options["formatters"] = get_formatters(get_option("formatter", yaml_config, default={}),
                                                  result_options["silent"])

    limit = get_option("limit", yaml_config, alt=app_options.limit)
    if limit:
        result_options["limit"] = int(limit)

    return core.main(**result_options)


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-t", "--task", dest="task_id",
                      help="redmine task identifier")
    parser.add_option("-c", "--commit", dest="commit", default=[], type=str, action='append',
                      help="SHA of commit")
    parser.add_option("--min_weight", dest="min_weight", default=0.1,
                      help=u'Минимальный вес возвращаемых тасков.')
    parser.add_option("--min_impact_rate", dest="min_impact_rate", default=0.15,
                      help=u'Если кол-во затронутых тасков для файла больше 15% от общего числа тасков, то пропускаем. 1 – все файлы, 0 – без файлов')
    parser.add_option("--all_commits", dest="check_all_commits", action="store_true", default=False,
                      help=u'By default gitimpact checks only child commits')
    parser.add_option("--silent", dest="silent", action="store_true", default=False,
                      help=u'Print only formatted result')
    parser.add_option("-n", dest="limit", default=None,
                      help=u'Result issues count')
    parser.add_option("--config", dest="yaml_config_file", default=os.path.join(os.getcwd(), ".gitimpact.yml"),
                      help=u'Path to yaml configuration file')
    parser.add_option("--repo", dest="repo_path", default=os.getcwd(),
                      help=u'Path to git repository')

    (options, _) = parser.parse_args()

    print main(options)
