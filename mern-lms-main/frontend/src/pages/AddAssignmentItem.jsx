import { useDispatch, useSelector } from "react-redux";
import { useEffect, useState } from "react";
import { Link, useLocation, useNavigate, useParams } from "react-router-dom";
import {
  getDownloadURL,
  getStorage,
  ref,
  uploadBytesResumable,
} from "firebase/storage";
import { app } from "../firebase";
import ClassSidebar from "../components/ClassSidebar";
import {
  Accordion,
  AccordionSummary,
  Alert,
  Button,
  Typography,
} from "@mui/material";
import { Label, Modal, Spinner, Textarea } from "flowbite-react";
import pdf from "../assets/pdf.png";
import moment from "moment";
import { toggleIsEditMode } from "../redux/isEditMode/isEditModeSlice";
import {
  Assignment as AssignmentIcon,
  ExpandMore as ExpandMoreIcon,
  Groups as GroupsIcon,
  Person as PersonIcon,
} from "@mui/icons-material";
import { MdOutlineCloudDone } from "react-icons/md";

export default function AddAssignmentItem() {
  const dispatch = useDispatch();
  const navigate = useNavigate();
  const { currentUser } = useSelector((state) => state.user);
  const { isEditMode } = useSelector((state) => state.isEditMode);
  const { classId } = useParams();
  const { tabIndex } = useSelector((state) => state.tabIndex);
  const location = useLocation();
  const [tabValue, setTabValue] = useState("");
  const [files, setFiles] = useState([]);
  const [nameFiles, setNameFiles] = useState([]);
  const [formData, setFormData] = useState({
    materialUrls: [],
    description: "",
  });
  const [filesUploadError, setFilesUploadError] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [createSuccess, setCreateSuccess] = useState(null);
  const [createError, setCreateError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [assignmentId, setAssignmentId] = useState("");
  const [assignment, setAssignment] = useState([]);
  const [classInfo, setClassInfo] = useState({});

  const [showModalCreateSuccess, setShowModalCreateSuccess] = useState(false);
  const [classImage, setClassImage] = useState("");
  const imageURLs = [
    "https://img.freepik.com/free-psd/realistic-school-supplies_23-2150588345.jpg",
    "https://img.freepik.com/free-vector/geometric-science-education-background-vector-gradient-blue-digital-remix_53876-125993.jpg",
    "https://img.freepik.com/free-vector/education-pattern-background-doodle-style_53876-115365.jpg",
    "https://img.freepik.com/free-vector/gradient-international-day-education-background_23-2151120677.jpg",
    "https://img.freepik.com/free-vector/hand-drawn-back-school-background_23-2149464866.jpg",
    "https://img.freepik.com/premium-photo/back-school-equipment-premium-psd_467500-32.jpg",
    "https://img.freepik.com/free-photo/desk-workspace-with-various-elements_23-2148043273.jpg",
    "https://img.freepik.com/free-photo/elevated-view-laptop-stationeries-blue-backdrop_23-2147880457.jpg",
    "https://img.freepik.com/free-photo/flat-lay-arrangement-desk-elements-with-copy-space_23-2148513316.jpg",
    "https://img.freepik.com/free-photo/blue-surface-with-study-tools_23-2147864592.jpg",
  ];

  useEffect(() => {
    if (tabIndex === 0) {
      setTabValue("Material");
    }
    if (tabIndex === 1) {
      setTabValue("Group");
    }
    if (tabIndex === 2) {
      setTabValue("Assignment");
    }
    if (tabIndex === 3) {
      setTabValue("Grade");
    }
    if (tabIndex === 4) {
      setTabValue("Forum");
    }
  }, [tabIndex]);

  useEffect(() => {
    const urlParams = new URLSearchParams(location.search);
    const tabFromUrl = urlParams.get("assignmentId");
    if (tabFromUrl) {
      setAssignmentId(tabFromUrl);
    }
    const fetchMaterials = async () => {
      if (assignmentId.length === 0) {
        return;
      }
      try {
        const resAssignment = await fetch(
          `/api/assignment/get/${classId}?assignmentId=${assignmentId}`
        );
        const assignment = await resAssignment.json();
        if (resAssignment.ok) {
          setAssignment(assignment.assignments[0]);
        }
      } catch (error) {
        console.log(error.message);
      }
    };
    fetchMaterials();
  }, [location.search, assignmentId, classId]);

  useEffect(() => {
    handleFilesSubmit();
    var arr = [];
    for (let i = 0; i < files.length; i++) {
      arr.push(files[i].name);
    }
    setNameFiles(arr);
  }, [files]);

  useEffect(() => {
    const stringToDecimal = (str) => {
      if (!str || typeof str !== "string") {
        return 1;
      }
      let sum = 0;
      for (let i = 0; i < str.length; i++) {
        sum += str.charCodeAt(i);
      }
      const normalized = sum % 1000 || 1;
      return Math.max(1, Math.floor(normalized));
    };
    const fetchClassInfo = async () => {
      try {
        const res = await fetch(`/api/class/get-info/${classId}`);
        const data = await res.json();

        if (res.ok) {
          setClassInfo(data);
          const uniqueString =
            data?.classItem?.name +
            data?.subject?.name +
            data?.subject?.code +
            data?.course?.startAcademicYear +
            data?.course?.endAcademicYear;
          const index = stringToDecimal(uniqueString) % imageURLs.length;
          setClassImage(imageURLs[index]);
        }
      } catch (error) {
        console.log(error.message);
      }
    };
    fetchClassInfo();
  }, [classId]);

  const handleFilesSubmit = (e) => {
    if (files.length > 0 && files.length + formData.materialUrls.length < 7) {
      setUploading(true);
      setFilesUploadError(false);
      const promises = [];
      for (let i = 0; i < files.length; i++) {
        promises.push(storeFiles(files[i]));
      }
      Promise.all(promises)
        .then((urls) => {
          setFormData({
            ...formData,
            materialUrls: urls,
          });
          setFilesUploadError(false);
          setUploading(false);
        })
        .catch((err) => {
          setFilesUploadError("File upload failed (20 MB max per file)");
          setUploading(false);
        });
    } else {
      setUploading(false);
    }
  };

  const storeFiles = async (file) => {
    return new Promise((resolve, reject) => {
      const storage = getStorage(app);
      const fileName = new Date().getTime() + file.name;
      const storageRef = ref(storage, fileName);
      const uploadTask = uploadBytesResumable(storageRef, file);
      uploadTask.on(
        "state_changed",
        (snapshot) => {
          const progress =
            (snapshot.bytesTransferred / snapshot.totalBytes) * 100;
          console.log(`Upload is ${progress}% done`);
        },
        (error) => {
          reject(error);
        },
        () => {
          getDownloadURL(uploadTask.snapshot.ref).then((downloadURL) => {
            resolve(downloadURL);
          });
        }
      );
    });
  };

  const handleRemoveFile = (index) => {
    setFormData({
      ...formData,
      materialUrls: formData.materialUrls.filter((_, i) => i !== index),
    });
    setNameFiles(nameFiles.filter((_, i) => i !== index));
  };

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.id]: e.target.value,
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setCreateSuccess(null);
    try {
      if (formData?.materialUrls?.length < 1)
        return setCreateError("You must upload at least one file");
      setLoading(true);
      setCreateError(null);
      const res = await fetch(
        `/api/material/create-material-assignment/${assignmentId}`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            ...formData,
            nameFiles,
            uploadBy: currentUser?._id,
          }),
        }
      );
      const data = await res.json();
      if (data.success === false) {
        setCreateError(data.message);
      } else {
        setCreateSuccess("Add assignment item successfully!");
        setShowModalCreateSuccess(true);
      }
      setLoading(false);
    } catch (error) {
      setCreateError(error.message);
      setLoading(false);
    }
  };
  return (
    <div className="min-h-screen flex flex-col mx-auto lg:w-10/12 mb-40">
      <div className="h-[240px] my-5 flex border-2 border-gray-300 rounded-xl overflow-hidden shadow-md">
        <img
          src={classImage}
          alt="class cover"
          className="w-[23vw] h-full border-r-2 border-gray-300"
        />
        <div className="p-12 flex flex-col gap-1 justify-center">
          <p className="text-xl font-bold">
            Semester {classInfo?.classItem?.semester % 10} | Academic Year{" "}
            {classInfo?.course?.startAcademicYear} -{" "}
            {classInfo?.course?.endAcademicYear}
          </p>
          <span className="text-3xl font-bold ">
            {classInfo?.subject?.code}_{classInfo?.subject?.name}
          </span>
          <div className="flex gap-2">
            <span className="text-xl text-cyan-600">Class:</span>
            <span className="text-xl text-gray-950">
              {classInfo?.classItem?.name}
            </span>
          </div>
          <div className="flex gap-2">
            <span className="text-xl text-cyan-600">Educators:</span>
            {classInfo?.educators?.length > 0 &&
              classInfo?.educators.map((educator, i) => (
                <span className="text-xl text-gray-950" key={i}>
                  {i !== classInfo?.educators?.length - 1
                    ? `${educator?.name},`
                    : educator?.name}
                </span>
              ))}
          </div>
          {currentUser?.isEducator && (
            <div className="flex gap-3 items-center">
              <span className="text-xl font-semibold">Edit Mode</span>
              {isEditMode ? (
                <i
                  className="fa-solid fa-toggle-on fa-xl cursor-pointer"
                  onClick={() => dispatch(toggleIsEditMode())}
                ></i>
              ) : (
                <i
                  className="fa-solid fa-toggle-off fa-xl cursor-pointer"
                  onClick={() => dispatch(toggleIsEditMode())}
                ></i>
              )}
            </div>
          )}
        </div>
      </div>
      <div className="flex flex-row">
        <div className="basis-9/12">
          <Accordion
            defaultExpanded
            className="mb-4"
            sx={{
              borderRadius: 2,
              overflow: "hidden",
            }}
          >
            <AccordionSummary
              expandIcon={<ExpandMoreIcon sx={{ color: "white" }} />}
              aria-controls="panel1-content"
              id="panel1-header"
              sx={{
                backgroundColor: "#26597C",
                color: "#ffffff",
                borderTopLeftRadius: 12,
                borderTopRightRadius: 12,
              }}
            >
              <div className="flex gap-2 items-center">
                <AssignmentIcon />
                <Typography
                  component="span"
                  style={{ fontSize: "18px", fontWeight: "bold" }}
                >
                  {assignment?.title}
                </Typography>
                {assignment?.type === "Individual" ? (
                  <PersonIcon />
                ) : (
                  <GroupsIcon />
                )}
              </div>
            </AccordionSummary>
            <div style={{ backgroundColor: "#F8F8D5", padding: "4px" }}>
              <div className="flex flex-col gap-2 my-5 lg:w-11/12 mx-auto">
                <div className="flex gap-2">
                  <div className="flex flex-col gap-2">
                    <div className="font-bold">Open Date: </div>
                    <div className="font-bold">Due Date: </div>
                    <div className="font-bold">Description: </div>
                  </div>
                  <div className="flex flex-col gap-2">
                    <div className="font-bold text-red-600">
                      {assignment?.startDate
                        ? moment(assignment?.startDate).format(
                            "HH:mm:ss DD/MM/YYYY"
                          )
                        : "---"}
                    </div>
                    <div className="font-bold text-red-600">
                      {assignment?.endDate
                        ? moment(assignment?.endDate).format(
                            "HH:mm:ss DD/MM/YYYY"
                          )
                        : "---"}
                    </div>
                    <div className="">{assignment?.description || "---"}</div>
                  </div>
                </div>
              </div>
            </div>
          </Accordion>
          <form onSubmit={handleSubmit} className="flex flex-col gap-4 mt-4">
            <div className="flex flex-col gap-1">
              <Label value="Description" className="text-lg" />
              <Textarea
                id="description"
                placeholder="Description"
                rows={8}
                maxLength="800"
                onChange={handleChange}
              />
            </div>
            <div className="flex flex-col flex-1 gap-4">
              <div className="flex gap-4">
                <input
                  onChange={(e) => setFiles(e.target.files)}
                  className="p-3 border border-gray-300 rounded w-full"
                  type="file"
                  id="files"
                  accept=".docx, .pdf"
                  multiple
                />
              </div>
              <p className="text-red-700 text-sm">
                {filesUploadError && filesUploadError}
              </p>
              {formData?.materialUrls?.length > 0 &&
                formData?.materialUrls.map((url, index) => (
                  <div
                    key={index}
                    className="flex justify-between p-3 border items-center"
                  >
                    <div className="flex items-center">
                      <img src={pdf} alt="pdf icon" className="w-6 h-6" />
                      <Link
                        to={url}
                        underline="hover"
                        target="_blank"
                        className="hover:underline text-cyan-600 ml-3"
                      >
                        {nameFiles[index]}
                      </Link>
                    </div>
                    <button
                      type="button"
                      onClick={() => handleRemoveFile(index)}
                      disabled={loading || uploading}
                      className="color-red"
                    >
                      <i className="fa-solid fa-trash hover:text-red-600"></i>
                    </button>
                  </div>
                ))}
              <div className="flex gap-2">
                <Button
                  variant="contained"
                  style={{
                    backgroundColor: "#26597C",
                    color: "#ffffff",
                    textTransform: "none",
                  }}
                  type="submit"
                  size="large"
                  disabled={loading || uploading}
                >
                  {uploading ? (
                    <>
                      <Spinner size="sm" />
                      <span className="pl-3">Uploading...</span>
                    </>
                  ) : loading ? (
                    <>
                      <Spinner size="sm" />
                      <span className="pl-3">Saving...</span>
                    </>
                  ) : (
                    "Save"
                  )}
                </Button>
                <Link to={`/class/${classId}`}>
                  <Button
                    variant="contained"
                    style={{
                      backgroundColor: "oklch(0.577 0.245 27.325)",
                      color: "#ffffff",
                      textTransform: "none",
                    }}
                    size="large"
                    disabled={loading || uploading}
                  >
                    Cancel
                  </Button>
                </Link>
              </div>
              {createError && (
                <Alert severity="error" className="mb-4 border border-red-600">
                  {createError}
                </Alert>
              )}
              {createSuccess && (
                <Alert
                  severity="success"
                  className="mb-4 border border-green-600"
                >
                  {createSuccess}
                </Alert>
              )}
            </div>
          </form>
        </div>
        <div className="basis-3/12 ml-5">
          <ClassSidebar classId={classId} />
        </div>
      </div>
      <Modal
        show={showModalCreateSuccess}
        onClose={() => setShowModalCreateSuccess(false)}
        popup
      >
        <Modal.Header />
        <Modal.Body>
          <div className="text-center">
            <MdOutlineCloudDone className="h-16 w-16 text-green-600 dark:text-gray-200 mb-4 mx-auto" />
            <h3 className="mb-6 text-2xl text-green-600 font-bold">
              {createSuccess}
            </h3>
            <div className="flex justify-center gap-4">
              <Button
                variant="contained"
                component="label"
                style={{
                  backgroundColor: "oklch(0.577 0.245 27.325)",
                  color: "#ffffff",
                  textTransform: "none",
                }}
                onClick={() => navigate(`/class/${classId}`)}
                fullWidth
              >
                Back to {tabValue} tab
              </Button>
              <Button
                variant="contained"
                component="label"
                style={{
                  backgroundColor: "#26597C",
                  color: "#ffffff",
                  textTransform: "none",
                }}
                onClick={() => {
                  window.location.reload();
                }}
                fullWidth
              >
                Continue to create
              </Button>
            </div>
          </div>
        </Modal.Body>
      </Modal>
    </div>
  );
}
