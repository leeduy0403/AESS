import express from "express";
import { verifyToken } from "../utils/verifyUser.js";
import {
  createSubject,
  deleteSubject,
  getSubjectsAdmin,
  updateSubject,
} from "../controllers/subject.controller.js";

const router = express.Router();

router.get("/get", verifyToken, getSubjectsAdmin);
router.post("/create", verifyToken, createSubject);
router.delete("/delete/:subjectId", verifyToken, deleteSubject);
router.put("/update/:subjectId", verifyToken, updateSubject);

export default router;
