import express from "express";
import { verifyToken } from "../utils/verifyUser.js";
import {
  createReply,
  deleteReply,
  editReply,
  getReplies,
} from "../controllers/reply.controller.js";

const router = express.Router();

router.post("/create", verifyToken, createReply);
router.put("/edit/:replyId", verifyToken, editReply);
router.delete("/delete/:questionId/:replyId", verifyToken, deleteReply);
router.get("/get-replies", verifyToken, getReplies);

export default router;
