// 浏览器端与 FastAPI 总线通信的逻辑已经直接内嵌在 index.html 里（轮询 /api/bus、
// 截图上传 /api/upload_frame、到位回调 /api/arm_ready）。
// 如果你后续把自己的 URDF 页面替换 index.html，把这几段 fetch 调用移植到你的页面里即可。
export {};
