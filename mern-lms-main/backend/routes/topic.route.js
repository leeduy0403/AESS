import express from "express";
import { verifyToken } from "../utils/verifyUser.js";
import {
  createTopic,
  deleteTopic,
  editTopic,
  getTopics,
} from "../controllers/topic.controller.js";

const router = express.Router();

router.post("/create", verifyToken, createTopic);
router.put("/edit/:topicId", verifyToken, editTopic);
router.delete("/delete/:forumId/:topicId", verifyToken, deleteTopic);
router.get("/get-topics", verifyToken, getTopics);

export default router;
