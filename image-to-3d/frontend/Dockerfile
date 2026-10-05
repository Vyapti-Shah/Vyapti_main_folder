FROM node:20-alpine AS vendor
WORKDIR /v
RUN npm init -y >/dev/null && npm i three@0.160.0

FROM nginx:alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY public/ /usr/share/nginx/html/
COPY --from=vendor /v/node_modules/three/build /usr/share/nginx/html/vendor/three/build
COPY --from=vendor /v/node_modules/three/examples/jsm /usr/share/nginx/html/vendor/three/examples/jsm
