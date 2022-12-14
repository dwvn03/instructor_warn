const express = require('express');
const app = express();
const http = require('http');
const server = http.createServer(app);
const { Server } = require("socket.io");

const io = new Server(server, {
    cors: { origin: "*" }
});

app.get('/', (req, res) => {
	res.send("<h1>Backend for instructor detection</h1>");
});

app.get('/warn', (req, res) => {
	console.log("pinged");
	io.emit('warned', "py" );
	res.send("");
});

io.on('connection', (socket) => {
    console.log('a user connected');

	socket.on('disconnect', () => {
		console.log('a user disconnected');
	});

    socket.on('message', (message) => {
        console.log(message);
        io.emit('message', `${socket.id} said ${message}` );   
    });

    socket.on('ping', (text) => {
        io.emit('warned', text ?? "" );   
    });
});

const PORT = process.env.PORT || 8080;

server.listen(PORT, () => {
  console.log('listening on *:8080');
});