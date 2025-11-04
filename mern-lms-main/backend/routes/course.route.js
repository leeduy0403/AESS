import express from "express";
import { verifyToken } from "../utils/verifyUser.js";
import {
  createCourse,
  createMultipleCourses,
  deleteCourse,
  getCoursesAdmin,
  updateCourse,
} from "../controllers/course.controller.js";

const router = express.Router();

router.post("/create", verifyToken, createCourse);
router.post("/create-multiple", verifyToken, createMultipleCourses);
router.get("/get", verifyToken, getCoursesAdmin);
router.delete("/delete/:courseId", verifyToken, deleteCourse);
router.put("/update/:courseId", verifyToken, updateCourse);

export default router;
