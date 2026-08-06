---
layout: default
title: 首页
body_class: home-page
---

<section class="profile-shell{% unless site.data.profile %} profile-only{% endunless %}" aria-labelledby="profile-name">
  <aside class="profile-intro">
    <p class="eyebrow">PERSONAL HOMEPAGE</p>
    <h1 id="profile-name">Yanlin Li</h1>
    <p class="profile-summary">个人简介待补充</p>

    <div class="profile-actions">
      <a class="button button-primary" href="{{ '/blog/' | relative_url }}">浏览博客</a>
      <a class="button button-secondary" href="{{ site.github_url }}" target="_blank" rel="noopener noreferrer">
        GitHub
        <span aria-hidden="true">↗</span>
      </a>
    </div>
  </aside>

  {% if site.data.profile %}
  <div class="profile-details">
    {% if site.data.profile.education %}
    <section class="profile-section">
      <h2>教育经历</h2>
      {% for item in site.data.profile.education %}
      <article class="experience-item">
        <div>
          <h3>{{ item.institution }}</h3>
          {% if item.detail %}<p>{{ item.detail }}</p>{% endif %}
        </div>
        {% if item.period %}<p class="experience-period">{{ item.period }}</p>{% endif %}
      </article>
      {% endfor %}
    </section>
    {% endif %}

    {% if site.data.profile.projects %}
    <section class="profile-section">
      <h2>项目</h2>
      <div class="project-list">
        {% for item in site.data.profile.projects %}
        <article class="project-item">
          <h3>
            {% if item.url %}<a href="{{ item.url }}" target="_blank" rel="noopener noreferrer">{{ item.name }}</a>
            {% else %}{{ item.name }}{% endif %}
          </h3>
          {% if item.description %}<p>{{ item.description }}</p>{% endif %}
        </article>
        {% endfor %}
      </div>
    </section>
    {% endif %}

    {% if site.data.profile.skills %}
    <section class="profile-section">
      <h2>技能</h2>
      <ul class="skill-list" aria-label="技能列表">
        {% for skill in site.data.profile.skills %}
        <li>{{ skill }}</li>
        {% endfor %}
      </ul>
    </section>
    {% endif %}
  </div>
  {% endif %}
</section>
