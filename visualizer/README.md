# FaceRec Visualizer

Web application for interacting with the system.

Features:
- see and manipulate the list of celebrities to search,
- include/exclude some web-crawled images from training set,
- train a model and see the results on video

Demo at http://facerec.eurecom.fr/visualizer

## Usage

Install:
```
npm install
```

Compiles and hot-reloads for development:
```
npm run serve
```

Compiles and minifies for production:
```
npm run build
```

Lints and fixes files
```
npm run lint
```

## Docker
```
docker build -t facerec/visualizer .
docker run -p 8080:80 --name facerec-visualizer --restart unless-stopped -d facerec/visualizer

# uninstall
docker stop facerec-visualizer
docker rm facerec-visualizer
docker rmi facerec/visualizer
```