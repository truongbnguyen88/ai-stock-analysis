import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import "./index.css";
// Mockup-exact component layer — loaded after index.css so its semantic classes win over Tailwind
// preflight (see web/src/styles/app.css, ported from the reference artifact).
import "./styles/app.css";

const root = document.getElementById("root");
if (!root) throw new Error("#root not found");

createRoot(root).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
