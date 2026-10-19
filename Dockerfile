# Build on the host first: npm ci && npm run models && npm run build
# This container serves the resulting static files; no Rust/Node runtime is shipped.
FROM nginx:stable-alpine
COPY deploy/nginx.conf /etc/nginx/conf.d/default.conf
COPY dist/ /usr/share/nginx/html/
EXPOSE 80
