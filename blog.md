---
layout: default
title: 博客
permalink: /blog/
body_class: blog-page
---

<header class="page-heading">
  <p class="section-kicker">BLOG</p>
  <h1>博客</h1>
  <p>学习过程、公式推导与实践记录。</p>
</header>

<section class="series-section" aria-labelledby="rl-foundations">
  <div class="series-heading">
    <div>
      <p class="series-label">SERIES 01</p>
      <h2 id="rl-foundations">强化学习基础</h2>
    </div>
    <p class="series-count">
      {% assign series_posts = site.posts | where: "series", "强化学习基础" %}
      {{ series_posts.size }} 篇博客
    </p>
  </div>

  <div class="post-grid">
    {% for post in series_posts %}
    <article class="post-card">
      <div class="post-card-body">
        <div class="post-card-meta">
          <time datetime="{{ post.date | date_to_xmlschema }}">{{ post.date | date: "%Y.%m.%d" }}</time>
          {% if post.tags %}
          <ul class="tag-list" aria-label="博客标签">
            {% for tag in post.tags limit: 3 %}
            <li>{{ tag }}</li>
            {% endfor %}
          </ul>
          {% endif %}
        </div>
        <h3><a href="{{ post.url | relative_url }}">{{ post.title }}</a></h3>
        <p>{{ post.excerpt | strip_html | normalize_whitespace | truncate: 150 }}</p>
      </div>
      <a class="post-arrow" href="{{ post.url | relative_url }}" aria-label="阅读《{{ post.title }}》">阅读全文 →</a>
    </article>
    {% else %}
    <div class="empty-state">系列内容正在整理中。</div>
    {% endfor %}
  </div>
</section>
