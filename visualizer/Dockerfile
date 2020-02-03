FROM node:lts-alpine

RUN npm install -g http-server

WORKDIR /app

COPY package*.json ./

RUN npm install

COPY . .
COPY vue.config.prod.js vue.config.js

RUN npm run build

EXPOSE 8080
CMD [ "http-server", "dist" ]
