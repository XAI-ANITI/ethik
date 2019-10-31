{%- extends 'basic.tpl' -%}
{% from 'mathjax.tpl' import mathjax %}


{%- block header -%}
{%- block html_head -%}
{% set nb_title = nb.metadata.get('title', '') or resources['metadata']['name'] %}
{% set widgets = "widgets" in nb.metadata %}

---
layout: notebook
title: "{{nb_title}}"
widgets: {{widgets}}
head: |
  {% for css in resources.inlining.css -%}
  <style type="text/css">
  {{ css|indent }}
  </style>
  {% endfor %}

js: |
  {{ mathjax() }}
---
{%- endblock html_head -%}
{%- endblock header -%}

{% block body %}
{{ '{% raw %}' }}
{{ super() }}
{{ '{% endraw %}' }}
{%- endblock body %}

{% block footer %}
{{ super() }}
{% endblock footer %}
