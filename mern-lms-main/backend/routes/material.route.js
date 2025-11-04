import express from "express";
import { verifyToken } from "../utils/verifyUser.js";
import {
  createMaterialAssignment,
  createMaterialSection,
  deleteMaterialAssignment,
  deleteMaterialSection,
  getMaterials,
  getMaterialsAssignment,
  getMaterialsSection,
  updateMaterial,
} from "../controllers/material.controller.js";

const router = express.Router();

router.get("/get-materials/:classId", verifyToken, getMaterials);
router.get(
  "/get-materials-section/:sectionId",
  verifyToken,
  getMaterialsSection
);
router.get(
  "/get-materials-assignment/:assignmentId",
  verifyToken,
  getMaterialsAssignment
);
router.post(
  "/create-material-section/:sectionId",
  verifyToken,
  createMaterialSection
);
router.post(
  "/create-material-assignment/:assignmentId",
  verifyToken,
  createMaterialAssignment
);
router.delete(
  "/delete-material-section/:sectionId/:materialId",
  verifyToken,
  deleteMaterialSection
);
router.delete(
  "/delete-material-assignment/:assignmentId/:materialId",
  verifyToken,
  deleteMaterialAssignment
);
router.put("/update/:materialId", verifyToken, updateMaterial);

export default router;
