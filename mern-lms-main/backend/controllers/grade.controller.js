import { errorHandler } from "../utils/error.js";
import Grade from "../models/grade.model.js";

export const create = async (req, res, next) => {
  if (!req.user.isAdmin && !req.user.isTeacher) {
    return next(
      errorHandler(403, "You are not allowed to create grade for student!")
    );
  }
  if (!req.body.gradeItem || !req.body.semester) {
    return next(errorHandler(403, "Please provide all required fields!"));
  }
  const newCourse = new Course({ ...req.body });
  try {
    const savedPost = await newCourse.save();
    res.status(200).json(savedPost);
  } catch (error) {
    next(error);
  }
};

export const getCourses = async (req, res, next) => {
  if (!req.user.isAdmin) {
    return next(errorHandler(403, "You are not allowed to get courses!"));
  }
  try {
    const startIndex = parseInt(req.query.startIndex) || 0;
    const limit = parseInt(req.query.limit) || 9;
    const sortDirection = req.query.order === "asc" ? 1 : -1;
    const courses = await Course.find({
      ...(req.query.code && { code: req.query.code }),
      ...(req.query.semester && {
        semester: { $regex: req.query.semester, $options: "i" },
      }),
      ...(req.query.courseId && { _id: req.query.courseId }),
      ...(req.query.searchTerm && {
        $or: [
          { code: { $regex: req.query.searchTerm, $options: "i" } },
          { name: { $regex: req.query.searchTerm, $options: "i" } },
        ],
      }),
    })
      .sort({ updatedAt: sortDirection })
      .skip(startIndex)
      .limit(limit);
    res.status(200).json(courses);
  } catch (error) {
    next(error);
  }
};

export const deleteCourse = async (req, res, next) => {
  if (!req.user.isAdmin) {
    return next(errorHandler(403, "You are not allowed to delete a course!"));
  }
  try {
    await Course.findByIdAndDelete(req.params.courseId);
    res.status(200).json("Course has been deleted!");
  } catch (error) {
    next(error);
  }
};

export const updateCourse = async (req, res, next) => {
  if (!req.user.isAdmin) {
    return next(errorHandler(403, "You are not allowed to get courses!"));
  }
  try {
    const updatedCourse = await Course.findByIdAndUpdate(
      req.params.courseId,
      {
        $set: {
          code: req.body.code,
          semester: req.body.semester,
        },
      },
      { new: true }
    );
    res.status(200).json(updatedCourse);
  } catch (error) {
    next(error);
  }
};
