{{ fullname | escape | underline}}

.. automodule:: {{ fullname }}

{{ 'Classes: Summary' }}
{{ '----------------' }}

{% block classes %}
{# To prevent template error, but template should not be applied to such modules #}
{% if classes %}
   {{- '.. autosummary::' }}
   {%- for item in classes %}
      {%- if not item.startswith('_') %}
      {{ item }}
      {%- endif %}
   {%- endfor %}
{% endif %}
{% endblock %}


{{ 'Classes: Descriptions'}}
{{ '---------------------' }}

{% block classes2 %}
{# To prevent template error, but template should not be applied to such modules #}
{% if classes %}
   {%- for item in classes %}
      {%- if not item.startswith('_') %}
.. autoclass:: {{ item }}
      {%- endif %}
   {%- endfor %}
{% endif %}
{% endblock %}