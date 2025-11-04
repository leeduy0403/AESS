import express from "express";
import mongoose from "mongoose";
import dotenv from "dotenv";
import userRoutes from "./routes/user.route.js";
import authRoutes from "./routes/auth.route.js";
import facultyRoutes from "./routes/faculty.route.js";
import subjectRoutes from "./routes/subject.route.js";
import courseRoutes from "./routes/course.route.js";
import classRoutes from "./routes/class.route.js";
import sectionRoutes from "./routes/section.route.js";
import assignmentRoutes from "./routes/assignment.route.js";
import materialRoutes from "./routes/material.route.js";
import groupRoutes from "./routes/group.route.js";
import submissionRoutes from "./routes/submission.route.js";
import forumRoutes from "./routes/forum.route.js";
import topicRoutes from "./routes/topic.route.js";
import questionRoutes from "./routes/question.route.js";
import replyRoutes from "./routes/reply.route.js";
import cookieParser from "cookie-parser";

import cors from "cors";
import setupSocket from "./socket.js";
import messageRoute from "./routes/message.route.js";
import contactRoute from "./routes/contact.route.js";
import channelRoute from "./routes/channel.route.js";

import path from "path";

dotenv.config();

mongoose
  .connect(process.env.MONGO)
  .then(() => {
    console.log("MongoDB is connected");
  })
  .catch((err) => {
    console.log(err);
  });

const __dirname = path.resolve();

const app = express();
app.use(express.json());
app.use(cookieParser());

const server = app.listen(3000, () => {
  console.log("Server is running on port 3000!");
});

// Add cross origin resource sharing
app.use(
  cors({
    origin: process.env.ORIGIN,
    methods: ["GET", "POST", "PUT", "PATCH", "DELETE"],
    credentials: true,
  })
);

app.use("/api/user", userRoutes);
app.use("/api/auth", authRoutes);
app.use("/api/faculty", facultyRoutes);
app.use("/api/subject", subjectRoutes);
app.use("/api/course", courseRoutes);
app.use("/api/class", classRoutes);
app.use("/api/section", sectionRoutes);
app.use("/api/assignment", assignmentRoutes);
app.use("/api/material", materialRoutes);
app.use("/api/group", groupRoutes);
app.use("/api/submission", submissionRoutes);
app.use("/api/forum", forumRoutes);
app.use("/api/topic", topicRoutes);
app.use("/api/question", questionRoutes);
app.use("/api/reply", replyRoutes);

app.use("/api/message", messageRoute);
app.use("/api/contact", contactRoute);
app.use("/api/channel", channelRoute);

app.use(express.static(path.join(__dirname, "../frontend/dist")));

app.get("*", (req, res) => {
  res.sendFile(path.join(__dirname, "../frontend/dist/index.html"));
});

// Setup server in socketio
setupSocket(server);

app.use((err, req, res, next) => {
  const statusCode = err.statusCode || 500;
  const message = err.message || "Internal Server Error";
  res.status(statusCode).json({
    success: false,
    statusCode,
    message,
  });
});
