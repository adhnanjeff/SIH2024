import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'; // Import Routes and Route
import Home from './pages/Home';
import Cameras from './pages/Cameras';

function App() {
  return (
    <Router>
      <Routes> {/* Replace Switch with Routes */}
        <Route path="/" element={<Home />} /> {/* Modify Route component usage */}
        <Route path="/cameras" element={<Cameras />} /> {/* Modify Route component usage */}
      </Routes>
    </Router>
  );
}

export default App;
