import { errorHandler } from "../utils/error.js";
import Course from "../models/course.model.js";

export const createCourse = async (req, res, next) => {
  if (!req.user.isAdmin) {
    return next(errorHandler(403, "You are not allowed to create a course!"));
  }
  if (
    !req.body.startAcademicYear ||
    !req.body.endAcademicYear ||
    !req.body.subjectId
  ) {
    return next(errorHandler(403, "Please provide all required fields!"));
  }
  const newCourse = new Course({ ...req.body });
  try {
    const savedCourse = await newCourse.save();
    res.status(200).json(savedCourse);
  } catch (error) {
    next(error);
  }
};

export const createMultipleCourses = async (req, res, next) => {
  if (!req.user.isAdmin) {
    return next(errorHandler(403, "You are not allowed to create courses!"));
  }
  const { courses } = req.body;
  if (!Array.isArray(courses) || courses.length === 0) {
    return next(errorHandler(400, "No courses provided!"));
  }
  try {
    const newCourses = courses.map((course) => ({
      subjectId: course.subjectId,
      startAcademicYear: course.startAcademicYear,
      endAcademicYear: course.endAcademicYear,
      semester: course.semester,
    }));
    const savedCourses = await Course.insertMany(newCourses);
    res.status(200).json(savedCourses);
  } catch (error) {
    next(error);
  }
};

export const getCoursesAdmin = async (req, res, next) => {
  if (!req.user.isAdmin) {
    return next(errorHandler(403, "You are not allowed to get classes!"));
  }
  try {
    const startIndex = parseInt(req.query.startIndex) || 0;
    const limit = parseInt(req.query.limit) || 9;
    const sortDirection = req.query.order === "asc" ? 1 : -1;
    const courses = await Course.find({})
      .sort({ updatedAt: sortDirection })
      .skip(startIndex)
      .limit(limit)
      .populate({
        path: "subjectId",
        model: "Subject",
        populate: {
          path: "facultyId",
          model: "Faculty",
        },
      });
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
    return next(errorHandler(403, "You are not allowed to update course!"));
  }
  try {
    const updatedCourse = await Course.findByIdAndUpdate(
      req.params.courseId,
      {
        $set: {
          semester: req.body.semester,
          startAcademicYear: req.body.startAcademicYear,
          endAcademicYear: req.body.endAcademicYear,
          subjectId: req.body.subjectId,
        },
      },
      { new: true }
    );
    res.status(200).json(updatedCourse);
  } catch (error) {
    next(error);
  }
};
