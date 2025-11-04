import express from "express";
import { verifyToken } from "../utils/verifyUser.js";
import {
  createForum,
  deleteForum,
  editForum,
  getForums,
} from "../controllers/forum.controller.js";

const router = express.Router();

router.post("/create", verifyToken, createForum);
router.put("/edit/:forumId", verifyToken, editForum);
router.delete("/delete/:classId/:forumId", verifyToken, deleteForum);
router.get("/get-forums/:classId", verifyToken, getForums);

export default router;
