import { errorHandler } from "../utils/error.js";
import Class from "../models/class.model.js";
import Course from "../models/course.model.js";
import Subject from "../models/subject.model.js";
import User from "../models/user.model.js";

export const createClass = async (req, res, next) => {
  if (!req.user.isAdmin) {
    return next(errorHandler(403, "You are not allowed to create a class!"));
  }
  if (!req.body.name || !req.body.courseId) {
    return next(errorHandler(403, "Please provide all required fields!"));
  }
  const newClass = new Class({ ...req.body });
  try {
    const savedClass = await newClass.save();
    res.status(200).json(savedClass);
  } catch (error) {
    next(error);
  }
};

export const createMultipleClasses = async (req, res, next) => {
  if (!req.user.isAdmin) {
    return next(errorHandler(403, "You are not allowed to create classes!"));
  }
  const { classes } = req.body;
  if (!Array.isArray(classes) || classes.length === 0) {
    return next(errorHandler(400, "No classes provided!"));
  }
  try {
    const newClasses = classes.map((classItem) => ({
      courseId: classItem.courseId,
      name: classItem.name,
    }));
    const savedClasses = await Class.insertMany(newClasses);
    res.status(200).json(savedClasses);
  } catch (error) {
    next(error);
  }
};

export const getClassesAdmin = async (req, res, next) => {
  if (!req.user.isAdmin) {
    return next(errorHandler(403, "You are not allowed to get classes!"));
  }
  try {
    const startIndex = parseInt(req.query.startIndex) || 0;
    const limit = parseInt(req.query.limit) || 9;
    const sortDirection = req.query.order === "asc" ? 1 : -1;
    const classes = await Class.find({})
      .sort({ updatedAt: sortDirection })
      .skip(startIndex)
      .limit(limit)
      .populate({
        path: "courseId",
        model: "Course",
        populate: {
          path: "subjectId",
          model: "Subject",
          populate: {
            path: "facultyId",
            model: "Faculty",
          },
        },
      });
    res.status(200).json(classes);
  } catch (error) {
    next(error);
  }
};

export const deleteClass = async (req, res, next) => {
  if (!req.user.isAdmin) {
    return next(errorHandler(403, "You are not allowed to delete a class!"));
  }
  try {
    await Class.findByIdAndDelete(req.params.classId);
    res.status(200).json("Class has been deleted!");
  } catch (error) {
    next(error);
  }
};

export const updateClass = async (req, res, next) => {
  if (!req.user.isAdmin) {
    return next(errorHandler(403, "You are not allowed to update class!"));
  }
  try {
    const updatedClass = await Class.findByIdAndUpdate(req.params.classId, {
      new: true,
    });
    res.status(200).json(updatedClass);
  } catch (error) {
    next(error);
  }
};

export const assign = async (req, res, next) => {
  if (!req.user.isAdmin) {
    return next(
      errorHandler(403, "You are not allowed to assign students into class!")
    );
  }
  if (!req.body.classIds || !req.body.educatorIds || !req.body.studentIds) {
    return next(errorHandler(403, "Please provide all required fields!"));
  }
  try {
    for (const classId of req.body.classIds) {
      const assignClass = await Class.findById(classId);
      assignClass.educators = [
        ...new Set([...assignClass.educators, ...req.body.educatorIds]),
      ];
      assignClass.students = [
        ...new Set([...assignClass.students, ...req.body.studentIds]),
      ];
      await assignClass.save();
      req.body.educatorIds.map(async (educatorId) => {
        const educator = await User.findById(educatorId);
        educator.classes = [...new Set([...educator.classes, classId])];
        await educator.save();
      });
      req.body.studentIds.map(async (studentId) => {
        const student = await User.findById(studentId);
        student.classes = [...new Set([...student.classes, classId])];
        await student.save();
      });
    }
    res.status(200).json({ message: "Assignment successful!" });
  } catch (error) {
    next(error);
  }
};

export const getClassesOfUser = async (req, res, next) => {
  if (req.user.id !== req.params.userId) {
    return next(errorHandler(403, "You are not allowed to get classes!"));
  }
  try {
    const startIndex = parseInt(req.query.startIndex) || 0;
    const limit = parseInt(req.query.limit) || 9;
    const user = await User.findById(req.params.userId);
    const classes = await Class.find({
      _id: { $in: user.classes },
      ...(req.query.classId && { _id: req.query.classId }),
      ...(req.query.searchTerm && {
        $or: [
          { subjectCode: { $regex: req.query.searchTerm, $options: "i" } },
          { subjectName: { $regex: req.query.searchTerm, $options: "i" } },
        ],
      }),
      ...(req.query.semester && {
        semester: { $regex: req.query.semester, $options: "i" },
      }),
    })
      .skip(startIndex)
      .limit(limit);
    res.status(200).json(classes);
  } catch (error) {
    next(error);
  }
};

export const getClassInfo = async (req, res, next) => {
  try {
    const classItem = await Class.findById(req.params.classId);
    const course = await Course.findById(classItem.courseId);
    const subject = await Subject.findById(course.subjectId);
    const educators = await User.find({ _id: { $in: classItem.educators } });
    res.status(200).json({ classItem, course, subject, educators });
  } catch (error) {
    next(error);
  }
};
