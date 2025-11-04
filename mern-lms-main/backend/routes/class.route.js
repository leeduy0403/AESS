import express from "express";
import { verifyToken } from "../utils/verifyUser.js";
import {
  assign,
  createClass,
  createMultipleClasses,
  deleteClass,
  getClassesAdmin,
  getClassesOfUser,
  getClassInfo,
  updateClass,
} from "../controllers/class.controller.js";

const router = express.Router();

router.post("/create", verifyToken, createClass);
router.post("/create-multiple", verifyToken, createMultipleClasses);
router.get("/get", verifyToken, getClassesAdmin);
router.delete("/delete/:classId", verifyToken, deleteClass);
router.put("/update/:classId", verifyToken, updateClass);
router.put("/assign", verifyToken, assign);
router.get("/get/:userId", verifyToken, getClassesOfUser);
router.get("/get-info/:classId", verifyToken, getClassInfo);

export default router;
