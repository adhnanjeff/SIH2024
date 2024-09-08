import React from 'react';

const Cameras = () => {
  return (
    <div>
      <h2>Live Camera Feed</h2>
      <img
        src="http://localhost:5000/video_feed"
        alt="Camera Feed"
        style={{ width: '100%', height: 'auto' }}
      />
    </div>
  );
};

export default Cameras;
