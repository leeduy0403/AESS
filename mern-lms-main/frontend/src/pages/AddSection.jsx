import { useEffect, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import ClassSidebar from "../components/ClassSidebar";
import { Alert, Button } from "@mui/material";
import { Label, Modal, Textarea, TextInput } from "flowbite-react";
import { useDispatch, useSelector } from "react-redux";
import { toggleIsEditMode } from "../redux/isEditMode/isEditModeSlice";
import { MdOutlineCloudDone } from "react-icons/md";

export default function AddSection() {
  const dispatch = useDispatch();
  const navigate = useNavigate();
  const { currentUser } = useSelector((state) => state.user);
  const { isEditMode } = useSelector((state) => state.isEditMode);
  const { tabIndex } = useSelector((state) => state.tabIndex);
  const [tabValue, setTabValue] = useState("");
  const { classId } = useParams();
  const [formData, setFormData] = useState({
    name: "",
    description: "",
  });
  const [createSuccess, setCreateSuccess] = useState(null);
  const [createError, setCreateError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [classInfo, setClassInfo] = useState([]);
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

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.id]: e.target.value,
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setCreateSuccess(null);
    setCreateError(null);
    setLoading(true);
    try {
      const res = await fetch(`/api/section/create/${classId}`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ ...formData }),
      });
      const data = await res.json();
      if (data.success === false) {
        setCreateError(data.message);
      } else {
        setCreateSuccess("Add section successfully!");
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
          <main className="pr-5 mx-auto">
            <form
              onSubmit={handleSubmit}
              className="flex flex-col sm:flex-row gap-4"
            >
              <div className="flex flex-col gap-4 flex-1">
                <div className="flex flex-col gap-1">
                  <Label value="Name" className="text-lg" />
                  <TextInput
                    required
                    sizing="lg"
                    placeholder="Name"
                    id="name"
                    onChange={handleChange}
                  />
                </div>
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
                    disabled={loading}
                  >
                    {loading ? "Saving..." : "Save"}
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
                      disabled={loading}
                    >
                      Cancel
                    </Button>
                  </Link>
                </div>
                {createError && (
                  <Alert
                    severity="error"
                    className="mb-4 border border-red-600"
                  >
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
