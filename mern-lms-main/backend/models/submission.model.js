import mongoose from "mongoose";

const submissionSchema = new mongoose.Schema(
  {
    description: {
      type: String,
    },
    submissionUrls: {
      type: Array,
      required: true,
      default: [],
    },
    nameFiles: {
      type: Array,
      required: true,
      default: [],
    },
    gradingStatus: {
      type: String,
    },
    isHidden: {
      type: Boolean,
      default: false,
    },
    reviewRequest: {
      type: String,
    },
    userRequests: {
      type: Array,
      default: [],
    },
    timeRequests: {
      type: Array,
      default: [],
    },
    score: {
      type: Array,
      default: [],
    },
    scoreComponent: {
      type: Array,
      default: [],
    },
    coefficients: {
      type: Array,
      default: [],
    },
    overallScore: {
      type: Number,
    },
    overallAIScore: {
      type: Number,
    },
    individualScores: {
      type: Object,
      default: {},
    },
    feedback: {
      type: String,
    },
    groupId: {
      type: String,
    },
    uploadBy: {
      type: String,
      required: true,
    },
  },
  { timestamps: true }
);

const Submission = mongoose.model("Submission", submissionSchema);

export default Submission;
