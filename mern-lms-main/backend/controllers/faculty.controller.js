import { errorHandler } from "../utils/error.js";
import Faculty from "../models/faculty.model.js";

export const getFacultiesAdmin = async (req, res, next) => {
  if (!req.user.isAdmin) {
    return next(errorHandler(403, "You are not allowed to get faculties!"));
  }
  try {
    const startIndex = parseInt(req.query.startIndex) || 0;
    const limit = parseInt(req.query.limit) || 9;
    const sortDirection = req.query.order === "asc" ? 1 : -1;
    const faculties = await Faculty.find({})
      .sort({ updatedAt: sortDirection })
      .skip(startIndex)
      .limit(limit);
    res.status(200).json(faculties);
  } catch (error) {
    next(error);
  }
};

export const createFaculty = async (req, res, next) => {
  if (!req.user.isAdmin) {
    return next(errorHandler(403, "You are not allowed to create a faculty!"));
  }
  if (!req.body.name) {
    return next(errorHandler(403, "Please provide all required fields!"));
  }
  const newFaculty = new Faculty({ ...req.body });
  try {
    const savedFaculty = await newFaculty.save();
    res.status(200).json(savedFaculty);
  } catch (error) {
    next(error);
  }
};

export const deleteFaculty = async (req, res, next) => {
  if (!req.user.isAdmin) {
    return next(errorHandler(403, "You are not allowed to delete a faculty!"));
  }
  try {
    await Faculty.findByIdAndDelete(req.params.facultyId);
    res.status(200).json("Faculty has been deleted!");
  } catch (error) {
    next(error);
  }
};

export const updateFaculty = async (req, res, next) => {
  if (!req.user.isAdmin) {
    return next(errorHandler(403, "You are not allowed to update faculty!"));
  }
  try {
    const updatedFaculty = await Faculty.findByIdAndUpdate(
      req.params.facultyId,
      {
        $set: {
          name: req.body.name,
        },
      },
      { new: true }
    );
    res.status(200).json(updatedFaculty);
  } catch (error) {
    next(error);
  }
};
