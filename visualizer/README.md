# visualizer

## Project setup
```
npm install
```

### Compiles and hot-reloads for development
```
npm run serve
```

### Compiles and minifies for production
```
npm run build
```

### Lints and fixes files
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

### Customize configuration
See [Configuration Reference](https://cli.vuejs.org/config/).
