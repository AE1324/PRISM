// react-router-dom provides client-side routing.
import { BrowserRouter, Routes, Route } from "react-router-dom";
import HomePage from "./pages/home-page";
import MeasuresPage from "./pages/measures-page";
import MethodologyPage from "./pages/methodology-page";
import UploadPage from "./pages/upload-page";
import SamplePage from "./pages/sample-page";
import DashboardPage from "./pages/dashboard-page";

function App() {
  return (
    <BrowserRouter>
      <Routes>
        {/* Public site */}
        <Route path="/" element={<HomePage />} />
        <Route path="/measures" element={<MeasuresPage />} />
        <Route path="/methodology" element={<MethodologyPage />} />
        {/* Workspace */}
        <Route path="/workspace" element={<UploadPage />} />
        <Route path="/sample" element={<SamplePage />} />
        <Route path="/dashboard" element={<DashboardPage />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
