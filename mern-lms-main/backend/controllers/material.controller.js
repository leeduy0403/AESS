import { errorHandler } from "../utils/error.js";
import Material from "../models/material.model.js";
import Class from "../models/class.model.js";
import Section from "../models/section.model.js";
import Assignment from "../models/assignment.model.js";

export const getMaterials = async (req, res, next) => {
  try {
    const classData = await Class.findById(req.params.classId);
    if (!classData) {
      return res.status(404).json({ message: "Class not found" });
    }
    const startIndex = parseInt(req.query.startIndex) || 0;
    const limit = parseInt(req.query.limit) || 5;
    const sortDirection = req.query.order === "asc" ? 1 : -1;
    const materials = await Material.find({
      sectionId: { $in: classData.sections },
    })
      .skip(startIndex)
      .limit(limit)
      .sort({ createdAt: sortDirection });
    res.status(200).json(materials);
  } catch (error) {
    next(error);
  }
};

export const getMaterialsSection = async (req, res, next) => {
  try {
    const section = await Section.findById(req.params.sectionId);
    const materials = await Material.find({
      _id: { $in: section.materials },
      ...(req.query.materialId && { _id: req.query.materialId }),
    });
    res.status(200).json(materials);
  } catch (error) {
    next(error);
  }
};

export const getMaterialsAssignment = async (req, res, next) => {
  try {
    const assignment = await Assignment.findById(req.params.assignmentId);
    const materials = await Material.find({
      _id: { $in: assignment.materials },
      ...(req.query.materialId && { _id: req.query.materialId }),
    });
    res.status(200).json(materials);
  } catch (error) {
    next(error);
  }
};

export const createMaterialSection = async (req, res, next) => {
  if (!req.body.materialUrls || !req.body.nameFiles || !req.body.uploadBy) {
    return next(errorHandler(403, "Please provide all required fields!"));
  }
  const newMaterial = new Material({
    ...req.body,
    sectionId: req.params.sectionId,
  });
  try {
    const section = await Section.findById(req.params.sectionId);
    section.materials.push(newMaterial._id);
    await section.save();
    const savedMaterial = await newMaterial.save();
    res.status(200).json(savedMaterial);
  } catch (error) {
    next(error);
  }
};

export const createMaterialAssignment = async (req, res, next) => {
  if (!req.body.materialUrls || !req.body.nameFiles || !req.body.uploadBy) {
    return next(errorHandler(403, "Please provide all required fields!"));
  }
  const newMaterial = new Material({
    ...req.body,
    assignmentId: req.params.assignmentId,
  });
  try {
    const assignment = await Assignment.findById(req.params.assignmentId);
    assignment.materials.push(newMaterial._id);
    await assignment.save();
    const savedMaterial = await newMaterial.save();
    res.status(200).json(savedMaterial);
  } catch (error) {
    next(error);
  }
};

export const deleteMaterialSection = async (req, res, next) => {
  try {
    const material = await Material.findById(req.params.materialId);
    const section = await Section.findById(req.params.sectionId);
    section.materials = section.materials.filter(
      (item) => item.toString() !== material._id.toString()
    );
    await section.save();
    await Material.findByIdAndDelete(req.params.materialId);
    res.status(200).json("Material has been deleted!");
  } catch (error) {
    next(error);
  }
};

export const deleteMaterialAssignment = async (req, res, next) => {
  try {
    const material = await Material.findById(req.params.materialId);
    const assignment = await Assignment.findById(req.params.assignmentId);
    assignment.materials = assignment.materials.filter(
      (item) => item.toString() !== material._id.toString()
    );
    await assignment.save();
    await Material.findByIdAndDelete(req.params.materialId);
    res.status(200).json("Material has been deleted!");
  } catch (error) {
    next(error);
  }
};

export const updateMaterial = async (req, res, next) => {
  // if (!req.user.isAdmin) {
  //   return next(errorHandler(403, "You are not allowed to update material!"));
  // }
  try {
    const updatedMaterial = await Material.findByIdAndUpdate(
      req.params.materialId,
      {
        $set: {
          description: req.body.description,
          isHidden: req.body.isHidden,
          materialUrls: req.body.materialUrls,
          nameFiles: req.body.nameFiles,
        },
      },
      { new: true }
    );
    res.status(200).json(updatedMaterial);
  } catch (error) {
    next(error);
  }
};
