import express from "express";
import { verifyToken } from "../utils/verifyUser.js";
import {
  createFaculty,
  deleteFaculty,
  getFacultiesAdmin,
  updateFaculty,
} from "../controllers/faculty.controller.js";

const router = express.Router();

router.get("/get", verifyToken, getFacultiesAdmin);
router.post("/create", verifyToken, createFaculty);
router.delete("/delete/:facultyId", verifyToken, deleteFaculty);
router.put("/update/:facultyId", verifyToken, updateFaculty);

export default router;
