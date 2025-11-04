import express from "express";
import {
  deleteUser,
  getEducators,
  getStudents,
  getUser,
  getUsers,
  signout,
  test,
  updateUser,
} from "../controllers/user.controller.js";
import { verifyToken } from "../utils/verifyUser.js";

const router = express.Router();

router.get("/test", test);
router.put("/update/:userId", verifyToken, updateUser);
router.delete("/delete/:userId", verifyToken, deleteUser);
router.post("/signout", signout);
router.get("/get-users", verifyToken, getUsers);
router.get("/get-educators", verifyToken, getEducators);
router.get("/get-students", verifyToken, getStudents);
router.get("/:userId", getUser);

export default router;
