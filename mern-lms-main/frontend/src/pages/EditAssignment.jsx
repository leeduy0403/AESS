import { useEffect, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import ClassSidebar from "../components/ClassSidebar";
import { Alert, Button, MenuItem, Select } from "@mui/material";
import dayjs from "dayjs";
import { DemoContainer, DemoItem } from "@mui/x-date-pickers/internals/demo";
import { LocalizationProvider } from "@mui/x-date-pickers/LocalizationProvider";
import { AdapterDayjs } from "@mui/x-date-pickers/AdapterDayjs";
import Radio from "@mui/material/Radio";
import RadioGroup from "@mui/material/RadioGroup";
import FormControlLabel from "@mui/material/FormControlLabel";
import FormControl from "@mui/material/FormControl";
import FormGroup from "@mui/material/FormGroup";
import Checkbox from "@mui/material/Checkbox";
import { Label, Modal, Spinner, Textarea, TextInput } from "flowbite-react";
import { MobileDateTimePicker } from "@mui/x-date-pickers/MobileDateTimePicker";
import {
  getDownloadURL,
  getStorage,
  ref,
  uploadBytesResumable,
} from "firebase/storage";
import { app } from "../firebase";
import pdf from "../assets/pdf.png";
import { toggleIsEditMode } from "../redux/isEditMode/isEditModeSlice";
import { useDispatch, useSelector } from "react-redux";
import { MdOutlineCloudDone } from "react-icons/md";

export default function EditAssignment() {
  const dispatch = useDispatch();
  const navigate = useNavigate();
  const { currentUser } = useSelector((state) => state.user);
  const { isEditMode } = useSelector((state) => state.isEditMode);
  const { tabIndex } = useSelector((state) => state.tabIndex);
  const [tabValue, setTabValue] = useState("");
  const { classId } = useParams();
  const { assignmentId } = useParams();
  const [updateSuccess, setUpdateSuccess] = useState(null);
  const [updateError, setUpdateError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [classInfo, setClassInfo] = useState([]);
  const [assignmentInfo, setAssignmentInfo] = useState([]);
  const [formData, setFormData] = useState({
    title: "",
    description: "",
    startDate: "",
    endDate: "",
    triggerDate: "",
    type: "",
    submissionFormat: [],
    allowModify: "",
    maxNumberOfFile: "",
    maxAttempt: "",
    totalFileSize: "",
    maxMemberGroup: "",
    startDateGroup: "",
    endDateGroup: "",
    descriptions: [],
    rubrics: [],
  });
  const [descriptionFiles, setDescriptionFiles] = useState([]);
  const [descriptionNameFiles, setDescriptionNameFiles] = useState([]);
  const [descriptionFilesUploadError, setDescriptionFilesUploadError] =
    useState(null);
  const [descriptionUploading, setDescriptionUploading] = useState(false);
  const [rubricFiles, setRubricFiles] = useState([]);
  const [rubricNameFiles, setRubricNameFiles] = useState([]);
  const [rubricFilesUploadError, setRubricFilesUploadError] = useState(null);
  const [rubricUploading, setRubricUploading] = useState(false);
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
    if (!assignmentInfo) return;
    setFormData({
      title: assignmentInfo?.title || "",
      description: assignmentInfo?.description || "",
      startDate: assignmentInfo?.startDate || "",
      endDate: assignmentInfo?.endDate || "",
      triggerDate: assignmentInfo?.triggerDate || "",
      type: assignmentInfo?.type || "",
      submissionFormat: assignmentInfo?.submissionFormat || [],
      allowModify:
        assignmentInfo?.allowModify !== undefined
          ? assignmentInfo?.allowModify
          : false,
      maxNumberOfFile: assignmentInfo?.maxNumberOfFile || "",
      maxAttempt: assignmentInfo?.maxAttempt || "",
      totalFileSize: assignmentInfo?.totalFileSize || "",
      maxMemberGroup: assignmentInfo?.maxMemberGroup || "",
      startDateGroup: assignmentInfo?.startDateGroup || "",
      endDateGroup: assignmentInfo?.endDateGroup || "",
      descriptions: assignmentInfo?.descriptions || [],
      rubrics: assignmentInfo?.rubrics || [],
    });
    setDescriptionNameFiles(assignmentInfo?.descriptionNameFiles || []);
    setRubricNameFiles(assignmentInfo?.rubricNameFiles || []);
  }, [assignmentInfo]);
  console.log(formData);

  useEffect(() => {
    handleDescriptionFilesSubmit();
    var arr = [];
    for (let i = 0; i < descriptionFiles?.length; i++) {
      arr.push(descriptionFiles[i]?.name);
    }
    setDescriptionNameFiles(arr);
  }, [descriptionFiles]);

  useEffect(() => {
    handleRubricFilesSubmit();
    var arr = [];
    for (let i = 0; i < rubricFiles?.length; i++) {
      arr.push(rubricFiles[i]?.name);
    }
    setRubricNameFiles(arr);
  }, [rubricFiles]);

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

  useEffect(() => {
    const fetchAssignmentInfo = async () => {
      try {
        const res = await fetch(
          `/api/assignment/get/${classId}?assignmentId=${assignmentId}`
        );
        const data = await res.json();
        if (res.ok) {
          setAssignmentInfo(data.assignments[0]);
        }
      } catch (error) {
        console.log(error.message);
      }
    };
    fetchAssignmentInfo();
  }, [classId, assignmentId]);

  const handleDescriptionFilesSubmit = (e) => {
    if (
      descriptionFiles?.length > 0 &&
      descriptionFiles?.length + formData?.descriptions?.length < 7
    ) {
      setDescriptionUploading(true);
      setDescriptionFilesUploadError(null);
      const promises = [];
      for (let i = 0; i < descriptionFiles?.length; i++) {
        promises.push(storeDescriptionFiles(descriptionFiles[i]));
      }
      Promise.all(promises)
        .then((urls) => {
          setFormData({
            ...formData,
            descriptions: urls,
          });
          setDescriptionFilesUploadError(null);
          setDescriptionUploading(false);
        })
        .catch((err) => {
          setDescriptionFilesUploadError(
            "File upload failed (20 MB max per file)"
          );
          setDescriptionUploading(false);
        });
    } else {
      setDescriptionUploading(false);
    }
  };

  const handleRubricFilesSubmit = (e) => {
    if (
      rubricFiles?.length > 0 &&
      rubricFiles?.length + formData?.rubrics?.length < 7
    ) {
      setRubricUploading(true);
      setRubricFilesUploadError(null);
      const promises = [];
      for (let i = 0; i < rubricFiles?.length; i++) {
        promises.push(storeRubricFiles(rubricFiles[i]));
      }
      Promise.all(promises)
        .then((urls) => {
          setFormData({
            ...formData,
            rubrics: urls,
          });
          setRubricFilesUploadError(null);
          setRubricUploading(false);
        })
        .catch((err) => {
          setRubricFilesUploadError("File upload failed (20 MB max per file)");
          setRubricUploading(false);
        });
    } else {
      setRubricUploading(false);
    }
  };

  const storeDescriptionFiles = async (file) => {
    setFormData({
      title: assignmentInfo?.title || "",
      description: assignmentInfo?.description || "",
      startDate: assignmentInfo?.startDate || "",
      endDate: assignmentInfo?.endDate || "",
      triggerDate: assignmentInfo?.triggerDate || "",
      type: assignmentInfo?.type || "",
      submissionFormat: assignmentInfo?.submissionFormat || [],
      allowModify: assignmentInfo?.allowModify || "",
      maxNumberOfFile: assignmentInfo?.maxNumberOfFile || "",
      maxAttempt: assignmentInfo?.maxAttempt || "",
      totalFileSize: assignmentInfo?.totalFileSize || "",
      maxMemberGroup: assignmentInfo?.maxMemberGroup || "",
      startDateGroup: assignmentInfo?.startDateGroup || "",
      endDateGroup: assignmentInfo?.endDateGroup || "",
      descriptions: assignmentInfo?.descriptions || [],
      rubrics: assignmentInfo?.rubrics || [],
    });
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

  const storeRubricFiles = async (file) => {
    setFormData({
      title: assignmentInfo?.title || "",
      description: assignmentInfo?.description || "",
      startDate: assignmentInfo?.startDate || "",
      endDate: assignmentInfo?.endDate || "",
      type: assignmentInfo?.type || "",
      submissionFormat: assignmentInfo?.submissionFormat || [],
      allowModify: assignmentInfo?.allowModify || "",
      maxNumberOfFile: assignmentInfo?.maxNumberOfFile || "",
      maxAttempt: assignmentInfo?.maxAttempt || "",
      totalFileSize: assignmentInfo?.totalFileSize || "",
      maxMemberGroup: assignmentInfo?.maxMemberGroup || "",
      startDateGroup: assignmentInfo?.startDateGroup || "",
      endDateGroup: assignmentInfo?.endDateGroup || "",
      descriptions: assignmentInfo?.descriptions || [],
      rubrics: assignmentInfo?.rubrics || [],
    });
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

  const handleRemoveDescriptionFile = (index) => {
    setFormData({
      ...formData,
      descriptions: formData.descriptions.filter((_, i) => i !== index),
    });
    setDescriptionNameFiles(descriptionNameFiles.filter((_, i) => i !== index));
  };

  const handleRemoveRubricFile = (index) => {
    setFormData({
      ...formData,
      rubrics: formData.rubrics.filter((_, i) => i !== index),
    });
    setRubricNameFiles(rubricNameFiles.filter((_, i) => i !== index));
  };

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.id]: e.target.value,
    });
  };

  const handleCheckboxChange = (event) => {
    const { checked, value } = event.target;
    setFormData((prev) => ({
      ...prev,
      submissionFormat: checked
        ? [...prev.submissionFormat, value]
        : prev.submissionFormat.filter((format) => format !== value),
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setUpdateSuccess(null);
    setUpdateError(null);
    setLoading(true);
    try {
      const res = await fetch(
        `/api/assignment/update/${classId}/${assignmentId}`,
        {
          method: "PUT",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            ...formData,
            descriptionNameFiles,
            rubricNameFiles,
          }),
        }
      );
      const data = await res.json();
      if (data.success === false) {
        setUpdateError(data.message);
      } else {
        setUpdateSuccess("Assignment updated successfully!");
        setShowModalCreateSuccess(true);
      }
      setLoading(false);
    } catch (error) {
      setUpdateError(error.message);
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
          <main className="pr-5 mx-auto">
            <form onSubmit={handleSubmit}>
              <div className="flex flex-col gap-5 flex-1">
                <div className="flex flex-col gap-1">
                  <Label value="Title" className="text-lg" />
                  <TextInput
                    required
                    sizing="lg"
                    placeholder="Title"
                    id="title"
                    value={formData?.title}
                    onChange={handleChange}
                  />
                </div>
                <div className="flex flex-col gap-1">
                  <Label value="Description" className="text-lg" />
                  <Textarea
                    id="description"
                    rows={8}
                    maxLength="800"
                    value={formData?.description}
                    onChange={handleChange}
                  />
                </div>
                <LocalizationProvider dateAdapter={AdapterDayjs}>
                  <DemoContainer components={["MobileDateTimePicker"]}>
                    <div className="flex gap-4">
                      <div className="flex flex-col gap-1 flex-1">
                        <Label
                          value="Assignment Start Date"
                          className="text-lg"
                        />
                        <DemoItem>
                          <MobileDateTimePicker
                            value={
                              formData?.startDate
                                ? dayjs(formData?.startDate)
                                : null
                            }
                            onChange={(newValue) => {
                              setFormData((prev) => ({
                                ...prev,
                                startDate: newValue
                                  ? newValue.toISOString()
                                  : "",
                              }));
                            }}
                            ampm={false}
                            maxDateTime={
                              formData?.endDate
                                ? dayjs(formData?.endDate)
                                : undefined
                            }
                            orientation={"landscape"}
                            onAccept={() => {
                              setTimeout(() => {
                                document.activeElement.blur();
                              }, 1);
                            }}
                            format="HH:mm DD/MM/YYYY"
                          />
                        </DemoItem>
                      </div>
                      <div className="flex flex-col gap-1 flex-1">
                        <Label
                          value="Assignment End Date"
                          className="text-lg"
                        />
                        <DemoItem>
                          <MobileDateTimePicker
                            value={
                              formData?.endDate
                                ? dayjs(formData?.endDate)
                                : null
                            }
                            onChange={(newValue) => {
                              setFormData((prev) => ({
                                ...prev,
                                endDate: newValue
                                  ? newValue?.toISOString()
                                  : "",
                              }));
                            }}
                            ampm={false}
                            minDateTime={
                              formData?.startDate
                                ? dayjs(formData?.startDate)
                                : undefined
                            }
                            orientation={"landscape"}
                            onAccept={() => {
                              setTimeout(() => {
                                document.activeElement.blur();
                              }, 1);
                            }}
                            format="HH:mm DD/MM/YYYY"
                          />
                        </DemoItem>
                      </div>
                    </div>
                  </DemoContainer>
                </LocalizationProvider>
                <LocalizationProvider dateAdapter={AdapterDayjs}>
                  <DemoContainer components={["MobileDateTimePicker"]}>
                    <div className="flex gap-4">
                      <div className="flex flex-col gap-1 flex-1">
                        <Label
                          value="Group Creation Start Date"
                          className="text-lg"
                        />
                        <DemoItem>
                          <MobileDateTimePicker
                            value={
                              formData?.startDateGroup
                                ? dayjs(formData?.startDateGroup)
                                : null
                            }
                            onChange={(newValue) => {
                              setFormData((prev) => ({
                                ...prev,
                                startDateGroup: newValue
                                  ? newValue?.toISOString()
                                  : "",
                              }));
                            }}
                            ampm={false}
                            minDateTime={
                              formData?.startDate
                                ? dayjs(formData?.startDate)
                                : undefined
                            }
                            maxDateTime={
                              formData?.endDateGroup
                                ? dayjs(formData?.endDateGroup)
                                : undefined
                            }
                            orientation={"landscape"}
                            onAccept={() => {
                              setTimeout(() => {
                                document.activeElement.blur();
                              }, 1);
                            }}
                            disabled={formData?.type !== "Group"}
                            sx={{
                              "& .MuiInputBase-root.Mui-disabled": {
                                backgroundColor: "#f0f0f0",
                              },
                            }}
                            format="HH:mm DD/MM/YYYY"
                          />
                        </DemoItem>
                      </div>
                      <div className="flex flex-col gap-1 flex-1">
                        <Label
                          value="Group Creation End Date"
                          className="text-lg"
                        />
                        <DemoItem>
                          <MobileDateTimePicker
                            value={
                              formData?.endDateGroup
                                ? dayjs(formData?.endDateGroup)
                                : null
                            }
                            onChange={(newValue) => {
                              setFormData((prev) => ({
                                ...prev,
                                endDateGroup: newValue
                                  ? newValue?.toISOString()
                                  : "",
                              }));
                            }}
                            ampm={false}
                            minDateTime={
                              formData?.startDateGroup
                                ? dayjs(formData?.startDateGroup)
                                : undefined
                            }
                            maxDateTime={
                              formData?.endDate
                                ? dayjs(formData?.endDate)
                                : undefined
                            }
                            orientation={"landscape"}
                            onAccept={() => {
                              setTimeout(() => {
                                document.activeElement.blur();
                              }, 1);
                            }}
                            disabled={formData?.type !== "Group"}
                            sx={{
                              "& .MuiInputBase-root.Mui-disabled": {
                                backgroundColor: "#f0f0f0",
                              },
                            }}
                            format="HH:mm DD/MM/YYYY"
                          />
                        </DemoItem>
                      </div>
                    </div>
                  </DemoContainer>
                </LocalizationProvider>
                <div className="flex gap-4">
                  <div className="flex-1">
                    <FormControl>
                      <Label value="Assignment Type" className="text-lg" />
                      <RadioGroup
                        row
                        aria-labelledby="type"
                        name="row-radio-buttons-group"
                        value={formData?.type}
                        onChange={(e) =>
                          setFormData((prev) => ({
                            ...prev,
                            type: e.target.value,
                          }))
                        }
                      >
                        <div className="flex-1">
                          <FormControlLabel
                            value="Group"
                            control={<Radio />}
                            label="Group"
                          />
                        </div>
                        <div className="flex-1">
                          <FormControlLabel
                            value="Individual"
                            control={<Radio />}
                            label="Individual"
                          />
                        </div>
                      </RadioGroup>
                    </FormControl>
                  </div>
                  <div className="flex flex-col gap-1 flex-1">
                    <Label
                      value="Maximum member of group"
                      className="text-lg"
                    />
                    <Select
                      value={
                        formData?.type !== "Group"
                          ? ""
                          : formData?.maxMemberGroup
                      }
                      onChange={(e) => {
                        setFormData({
                          ...formData,
                          maxMemberGroup: e.target.value,
                        });
                      }}
                      MenuProps={{
                        PaperProps: {
                          style: {
                            maxHeight: 37 * 5,
                          },
                        },
                      }}
                      disabled={formData?.type !== "Group"}
                      sx={{
                        "&.Mui-disabled .MuiSelect-select": {
                          backgroundColor: "#f0f0f0",
                        },
                      }}
                    >
                      {[...Array(19)].map((_, index) => (
                        <MenuItem key={index + 2} value={index + 2}>
                          {index + 2}
                        </MenuItem>
                      ))}
                    </Select>
                  </div>
                </div>
                <div className="flex gap-4">
                  <div className="flex-1">
                    <FormControl>
                      <Label
                        value="Allow multiple attempts"
                        className="text-lg"
                      />
                      <RadioGroup
                        row
                        aria-labelledby="allowModify"
                        name="row-radio-buttons-group"
                        value={formData?.allowModify}
                        onChange={(e) =>
                          setFormData((prev) => ({
                            ...prev,
                            allowModify: e.target.value,
                            maxAttempt:
                              e.target.value === "true"
                                ? formData?.maxAttempt
                                : 1,
                          }))
                        }
                      >
                        <div className="">
                          <FormControlLabel
                            value="true"
                            control={<Radio />}
                            label="Yes"
                          />
                        </div>
                        <div className="">
                          <FormControlLabel
                            value="false"
                            control={<Radio />}
                            label="No"
                          />
                        </div>
                      </RadioGroup>
                    </FormControl>
                  </div>
                  <div className="flex flex-col gap-1 flex-1">
                    <Label value="Maximum attempt" className="text-lg" />
                    <Select
                      value={
                        formData?.allowModify === true ||
                        formData?.allowModify === "true"
                          ? formData?.maxAttempt
                          : 1
                      }
                      onChange={(e) => {
                        setFormData({
                          ...formData,
                          maxAttempt: e.target.value,
                        });
                      }}
                      MenuProps={{
                        PaperProps: {
                          style: {
                            maxHeight: 37 * 5,
                          },
                        },
                      }}
                      disabled={
                        formData?.allowModify === false ||
                        formData?.allowModify === "false"
                      }
                      sx={{
                        "&.Mui-disabled .MuiSelect-select": {
                          backgroundColor: "#f0f0f0",
                        },
                      }}
                    >
                      {[...Array(20)].map((_, index) => (
                        <MenuItem key={index + 1} value={index + 1}>
                          {index + 1}
                        </MenuItem>
                      ))}
                    </Select>
                  </div>
                </div>
                <div className="flex gap-4">
                  <div className="flex-1">
                    <Label value="Submission Format" className="text-lg" />
                    <FormGroup>
                      <div className="flex">
                        <FormControlLabel
                          control={
                            <Checkbox
                              checked={formData?.submissionFormat?.includes(
                                ".pdf"
                              )}
                              onChange={handleCheckboxChange}
                              value=".pdf"
                            />
                          }
                          label=".pdf"
                        />
                        <FormControlLabel
                          control={
                            <Checkbox
                              checked={formData?.submissionFormat?.includes(
                                ".docx"
                              )}
                              onChange={handleCheckboxChange}
                              value=".docx"
                            />
                          }
                          label=".docx"
                        />
                        <FormControlLabel
                          control={
                            <Checkbox
                              checked={formData?.submissionFormat?.includes(
                                ".txt"
                              )}
                              onChange={handleCheckboxChange}
                              value=".txt"
                            />
                          }
                          label=".txt"
                        />
                        <FormControlLabel
                          control={
                            <Checkbox
                              checked={formData?.submissionFormat?.includes(
                                ".md"
                              )}
                              onChange={handleCheckboxChange}
                              value=".md"
                            />
                          }
                          label=".md"
                        />
                      </div>
                    </FormGroup>
                  </div>
                  <div className="flex flex-col gap-1 flex-1">
                    <Label
                      value="Maximum file size (in MB)"
                      className="text-lg"
                    />
                    <Select
                      value={formData?.totalFileSize}
                      onChange={(e) => {
                        setFormData({
                          ...formData,
                          totalFileSize: e.target.value,
                        });
                      }}
                      MenuProps={{
                        PaperProps: {
                          style: {
                            maxHeight: 37 * 5,
                          },
                        },
                      }}
                    >
                      {[...Array(20)].map((_, index) => (
                        <MenuItem key={index + 1} value={index + 1}>
                          {index + 1}
                        </MenuItem>
                      ))}
                    </Select>
                  </div>
                </div>
                <LocalizationProvider dateAdapter={AdapterDayjs}>
                  <DemoContainer components={["MobileDateTimePicker"]}>
                    <div className="flex gap-4">
                      <div className="flex flex-col gap-1 flex-1">
                        <Label
                          value="Maximum number of submission files"
                          className="text-lg"
                        />
                        <Select
                          value={formData?.maxNumberOfFile}
                          onChange={(e) => {
                            setFormData({
                              ...formData,
                              maxNumberOfFile: e.target.value,
                            });
                          }}
                          MenuProps={{
                            PaperProps: {
                              style: {
                                maxHeight: 37 * 5,
                              },
                            },
                          }}
                        >
                          {[...Array(20)].map((_, index) => (
                            <MenuItem key={index + 1} value={index + 1}>
                              {index + 1}
                            </MenuItem>
                          ))}
                        </Select>
                      </div>
                      <div className="flex flex-col gap-1 flex-1">
                        <Label
                          value="Auto Evaluation Start Date"
                          className="text-lg"
                        />
                        <DemoItem>
                          <MobileDateTimePicker
                            value={
                              formData?.triggerDate
                                ? dayjs(formData?.triggerDate)
                                : null
                            }
                            onChange={(newValue) => {
                              setFormData((prev) => ({
                                ...prev,
                                triggerDate: newValue
                                  ? newValue?.toISOString()
                                  : "",
                              }));
                            }}
                            ampm={false}
                            minDateTime={
                              formData?.endDate
                                ? dayjs(formData?.endDate)
                                : undefined
                            }
                            orientation={"landscape"}
                            onAccept={() => {
                              setTimeout(() => {
                                document.activeElement.blur();
                              }, 1);
                            }}
                            format="HH:mm DD/MM/YYYY"
                          />
                        </DemoItem>
                      </div>
                    </div>
                  </DemoContainer>
                </LocalizationProvider>
                <div className="flex gap-4">
                  <div className="flex flex-col gap-1 flex-1">
                    <Label
                      value="Upload assignment description"
                      className="text-lg"
                    />
                    <input
                      onChange={(e) => setDescriptionFiles(e.target.files)}
                      className="p-3 border border-gray-300 rounded w-full"
                      type="file"
                      id="files"
                      accept=".docx, .pdf, .xlsx"
                      multiple
                      disabled={rubricUploading}
                    />
                  </div>
                  <div className="flex flex-col gap-1 flex-1">
                    <div className="flex items-center">
                      <Label
                        value="Upload assignment rubric"
                        className="text-lg"
                      />
                      <a
                        className="text-lg ml-auto text-cyan-600 hover:underline"
                        target="_blank"
                        href="https://drive.google.com/drive/folders/1U_BHw3kOeg-hlMfyH6Lzow9rb45YuJPn?usp=sharing"
                      >
                        Example rubrics
                      </a>
                    </div>
                    <input
                      onChange={(e) => setRubricFiles(e.target.files)}
                      className="p-3 border border-gray-300 rounded w-full"
                      type="file"
                      id="files"
                      accept=".docx, .pdf, .xlsx"
                      disabled={descriptionUploading}
                    />
                  </div>
                </div>
                <div className="flex gap-4">
                  <div className="flex-1">
                    <p className="text-red-700 text-sm">
                      {descriptionFilesUploadError &&
                        descriptionFilesUploadError}
                    </p>
                    {formData?.descriptions?.length > 0 &&
                      formData?.descriptions.map((url, index) => (
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
                              {descriptionNameFiles[index]}
                            </Link>
                          </div>
                          <button
                            type="button"
                            onClick={() => handleRemoveDescriptionFile(index)}
                            disabled={loading || descriptionUploading}
                            className="color-red"
                          >
                            <i className="fa-solid fa-trash hover:text-red-600"></i>
                          </button>
                        </div>
                      ))}
                  </div>
                  <div className="flex-1">
                    <p className="text-red-700 text-sm">
                      {rubricFilesUploadError && rubricFilesUploadError}
                    </p>
                    {formData?.rubrics?.length > 0 &&
                      formData?.rubrics.map((url, index) => (
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
                              {rubricNameFiles[index]}
                            </Link>
                          </div>
                          <button
                            type="button"
                            onClick={() => handleRemoveRubricFile(index)}
                            disabled={loading || rubricUploading}
                            className="color-red"
                          >
                            <i className="fa-solid fa-trash hover:text-red-600"></i>
                          </button>
                        </div>
                      ))}
                  </div>
                </div>
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
                    disabled={
                      loading || descriptionUploading || rubricUploading
                    }
                  >
                    {descriptionUploading || rubricUploading ? (
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
                      disabled={
                        loading || descriptionUploading || rubricUploading
                      }
                    >
                      Cancel
                    </Button>
                  </Link>
                </div>
                {updateError && (
                  <Alert
                    severity="error"
                    className="mb-4 border border-red-600"
                  >
                    {updateError}
                  </Alert>
                )}
                {updateSuccess && (
                  <Alert
                    severity="success"
                    className="mb-4 border border-green-600"
                  >
                    {updateSuccess}
                  </Alert>
                )}
              </div>
            </form>
          </main>
        </div>
        <div className="basis-3/12">
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
              {updateSuccess}
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
                Continue to update
              </Button>
            </div>
          </div>
        </Modal.Body>
      </Modal>
    </div>
  );
}
