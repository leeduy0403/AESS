import { errorHandler } from "../utils/error.js";
import Section from "../models/section.model.js";
import Material from "../models/material.model.js";
import Class from "../models/class.model.js";

export const getSections = async (req, res, next) => {
  try {
    const classItem = await Class.findById(req.params.classId);
    const sections = await Section.find({
      _id: { $in: classItem.sections },
      ...(req.query.sectionId && { _id: req.query.sectionId }),
    }).populate({
      path: "materials",
      model: "Material",
    });
    res.status(200).json(sections);
  } catch (error) {
    next(error);
  }
};

export const createSection = async (req, res, next) => {
  const classItem = await Class.findById(req.params.classId);
  if (!classItem.educators.includes(req.user.id)) {
    return next(
      errorHandler(
        403,
        "You are not allowed to create a section for this class!"
      )
    );
  }
  if (!req.body.name) {
    return next(errorHandler(403, "Please provide all required fields!"));
  }
  const newSection = new Section({ ...req.body });
  try {
    classItem.sections.push(newSection._id);
    await classItem.save();
    const savedSection = await newSection.save();
    res.status(200).json(savedSection);
  } catch (error) {
    next(error);
  }
};

export const deleteSection = async (req, res, next) => {
  const classItem = await Class.findById(req.params.classId);
  if (!classItem.educators.includes(req.user.id)) {
    return next(
      errorHandler(
        403,
        "You are not allowed to delete a section for this class!"
      )
    );
  }
  try {
    classItem.sections = classItem.sections.filter(
      (item) => item !== req.params.sectionId
    );
    await classItem.save();
    const section = await Section.findById(req.params.sectionId);
    const materials = await Material.find({
      _id: { $in: section.materials },
    });
    await Promise.all(
      materials.map(async (material) => {
        await Material.findByIdAndDelete(material._id);
      })
    );
    await Section.findByIdAndDelete(req.params.sectionId);
    res.status(200).json("Section has been deleted!");
  } catch (error) {
    next(error);
  }
};

export const updateSection = async (req, res, next) => {
  const classItem = await Class.findById(req.params.classId);
  if (!classItem.educators.includes(req.user.id)) {
    return next(
      errorHandler(
        403,
        "You are not allowed to update a section for this class!"
      )
    );
  }
  try {
    const updatedSection = await Section.findByIdAndUpdate(
      req.params.sectionId,
      {
        $set: {
          name: req.body.name,
          isHidden: req.body.isHidden,
          description: req.body.description,
          materials: req.body.materials,
        },
      },
      { new: true }
    );
    res.status(200).json(updatedSection);
  } catch (error) {
    next(error);
  }
};
