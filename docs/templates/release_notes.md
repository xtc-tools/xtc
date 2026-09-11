## XTC Release {{ tag }}

We're pleased to announce XTC version {{ tag }}. Thanks to all contributors.

## Important features

{% for pull in important_features_since_last_tag %}
* [{{ pull.title }}]({{ pull.url }})
  {{ pull.three_line_summary }}
{% endfor %}

## Important fixes

{% for pull in important_fixes_since_last_tag %}
* [{{ pull.title }}]({{ pull.url }}){% if pull.related_issue %}, fixes [{{ pull.related_issue.title }}]({{ pull.related_issue.url }}){% endif %}
  {{ pull.three_line_summary }}
{% endfor %}

## Changes

{% for pull in pulls_since_last_tag %}
* [{{ pull.title }}]({{ pull.url }}) by @{{ pull.user }}
{% endfor %}

## New contributors

{% for pull in new_contributor_first_pull %}
* @{{ pull.user }} made their first contribution in [{{ pull.title }}]({{ pull.url }})
{% endfor %}
