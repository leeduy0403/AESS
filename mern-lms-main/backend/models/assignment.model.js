import mongoose from "mongoose";

const assignmentSchema = new mongoose.Schema(
  {
    classId: {
      type: String,
    },
    title: {
      type: String,
      required: true,
    },
    description: {
      type: String,
    },
    startDate: {
      type: Date,
    },
    endDate: {
      type: Date,
    },
    triggerDate: {
      type: Date,
    },
    isTrigger: {
      type: Boolean,
      default: false,
    },
    type: {
      type: String,
    },
    status: {
      type: String,
    },
    isHidden: {
      type: Boolean,
      default: false,
    },
    isScorePublish: {
      type: Boolean,
      default: false,
    },
    isReviewRequest: {
      type: Boolean,
      default: false,
    },
    publishDate: {
      type: Date,
    },
    allowModify: {
      type: Boolean,
      default: true,
    },
    autoEvaluate: {
      type: Boolean,
      default: true,
    },
    maxNumberOfFile: {
      type: Number,
      default: 3,
    },
    maxAttempt: {
      type: Number,
      default: 3,
    },
    submissionFormats: {
      type: Array,
      default: [],
    },
    totalFileSize: {
      type: Number,
      default: 25,
    },
    gradingStatus: {
      type: String,
    },
    maxMemberGroup: {
      type: Number,
    },
    startDateGroup: {
      type: Date,
    },
    endDateGroup: {
      type: Date,
    },
    groups: {
      type: Array,
      default: [],
    },
    materials: {
      type: Array,
      default: [],
    },
    descriptions: {
      type: Array,
      default: [],
    },
    descriptionNameFiles: {
      type: Array,
      default: [],
    },
    rubrics: {
      type: Array,
      default: [],
    },
    rubricNameFiles: {
      type: Array,
      default: [],
    },
    submissions: {
      type: Array,
      default: [],
    },
  },
  { timestamps: true }
);

const Assignment = mongoose.model("Assignment", assignmentSchema);

export default Assignment;
