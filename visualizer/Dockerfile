FROM node:lts-alpine

RUN apk add --update nginx

# Create the directories we will need
RUN mkdir -p /var/log/nginx

# Copy the respective nginx configuration files
COPY nginx.conf /etc/nginx/nginx.conf
COPY default.conf /etc/nginx/conf.d/default.conf

WORKDIR /app

COPY package*.json ./

RUN npm install

COPY . ./
COPY app.config.prod.js app.config.js

RUN npm run build

RUN chown nginx:nginx /app/dist


EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
