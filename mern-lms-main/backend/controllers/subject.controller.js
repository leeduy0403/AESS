import { errorHandler } from "../utils/error.js";
import Subject from "../models/subject.model.js";

export const createSubject = async (req, res, next) => {
  if (!req.user.isAdmin) {
    return next(errorHandler(403, "You are not allowed to create a subject!"));
  }
  if (!req.body.name || !req.body.code || !req.body.facultyId) {
    return next(errorHandler(403, "Please provide all required fields!"));
  }
  const newSubject = new Subject({ ...req.body });
  try {
    const savedSubject = await newSubject.save();
    res.status(200).json(savedSubject);
  } catch (error) {
    next(error);
  }
};

export const getSubjectsAdmin = async (req, res, next) => {
  if (!req.user.isAdmin) {
    return next(errorHandler(403, "You are not allowed to get subjects!"));
  }
  try {
    const startIndex = parseInt(req.query.startIndex) || 0;
    const limit = parseInt(req.query.limit) || 9;
    const sortDirection = req.query.order === "asc" ? 1 : -1;
    const subjects = await Subject.find({})
      .sort({ updatedAt: sortDirection })
      .skip(startIndex)
      .limit(limit)
      .populate({
        path: "facultyId",
        model: "Faculty",
      });
    res.status(200).json(subjects);
  } catch (error) {
    next(error);
  }
};

export const deleteSubject = async (req, res, next) => {
  if (!req.user.isAdmin) {
    return next(errorHandler(403, "You are not allowed to delete a subject!"));
  }
  try {
    await Subject.findByIdAndDelete(req.params.subjectId);
    res.status(200).json("Subject has been deleted!");
  } catch (error) {
    next(error);
  }
};

export const updateSubject = async (req, res, next) => {
  if (!req.user.isAdmin) {
    return next(errorHandler(403, "You are not allowed to update subject!"));
  }
  try {
    const updatedSubject = await Subject.findByIdAndUpdate(
      req.params.subjectId,
      {
        $set: {
          name: req.body.name,
          code: req.body.code,
          facultyId: req.body.facultyId,
        },
      },
      { new: true }
    );
    res.status(200).json(updatedSubject);
  } catch (error) {
    next(error);
  }
};
