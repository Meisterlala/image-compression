# syntax=docker/dockerfile:1.19
# Build on the host first: npm ci && npm run models && npm run build
# This container serves the resulting static files; no Rust/Node runtime is shipped.
FROM nginx:stable-alpine
COPY deploy/nginx.conf /etc/nginx/conf.d/default.conf
# Keep large, rarely changing model bundles independent of frontend rebuilds.
COPY dist/models/ /usr/share/nginx/html/models/
COPY --exclude=models dist/ /usr/share/nginx/html/
EXPOSE 80
