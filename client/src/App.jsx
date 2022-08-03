import { useState, useEffect } from "react";
import io from "socket.io-client";

import { newNotif, askNotificationPermission } from "./notif";

const BASE_URL = import.meta.env.VITE_BASE_URL;
const socket = io(BASE_URL);

console.log(BASE_URL)
function App() {
  const [isConnected, setIsConnected] = useState(socket.connected);
  const [lastPong, setLastPong] = useState(null);
  const [value, setValue] = useState("");

  useEffect(() => {
    askNotificationPermission();

    socket.on("connect", () => {
      setIsConnected(true);
    });

    socket.on("disconnect", () => {
      setIsConnected(false);
    });

    socket.on("warned", (text) => {
      setLastPong(new Date().toISOString() + " " + text);
      newNotif("Be Careful", "Your instructor is looking at you");
    });

    return () => {
      socket.off("connect");
      socket.off("disconnect");
      socket.off("warned");
    };
  }, []);

  const sendPing = () => {
    socket.emit("ping");
  }

  const sendPingWithText = () => {
    socket.emit("ping", value);
  }

  return (
    <div>
      <p>Connected: { "" + isConnected }</p>
      <p>Last pong: { lastPong || "-" }</p>
      <button onClick={ sendPing }>Send ping</button>
      <input type="text" onChange={(e) => setValue(e.target.value) } />
      <button onClick={ sendPingWithText }>Ping text</button>
    </div>
  );
}

export default App;