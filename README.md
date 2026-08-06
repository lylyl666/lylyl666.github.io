# Yanlin Li — GitHub Pages

个人主页与系列博客，使用 Jekyll 和 GitHub Pages 构建。

## 页面

- `/`：个人主页
- `/blog/`：系列博客
- `/posts/.../`：博客正文

## 本地预览

```bash
bundle install
bundle exec jekyll serve
```

访问 `http://127.0.0.1:4000/`。

在带端口代理的开发环境中，使用：

```bash
./scripts/preview-proxy.sh
```

该脚本会使用代理路径构建网站，避免 CSS、图片和内部链接因根路径错误而返回 404。

## 发布策略

- `main`：当前线上版本
- `site-redesign`：新版个人主页和博客设计

设计确认前不要将 `site-redesign` 合并到 `main`。
