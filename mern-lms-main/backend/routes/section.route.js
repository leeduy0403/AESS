import express from "express";
import { verifyToken } from "../utils/verifyUser.js";
import {
  createSection,
  deleteSection,
  getSections,
  updateSection,
} from "../controllers/section.controller.js";

const router = express.Router();

router.get("/get/:classId", verifyToken, getSections);
router.post("/create/:classId", verifyToken, createSection);
router.delete("/delete/:classId/:sectionId", verifyToken, deleteSection);
router.put("/update/:classId/:sectionId", verifyToken, updateSection);

export default router;
