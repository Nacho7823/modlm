import { BrowserRouter, Routes, Route } from "react-router-dom";
import { ChatUI, Settings } from "./components";

function App() {
  return (
    <BrowserRouter>
      <div className="app">
        <Routes>
          <Route path="/" element={<ChatUI />} />
          <Route path="/settings" element={<Settings />} />
        </Routes>
      </div>
    </BrowserRouter>
  );
}

export default App;